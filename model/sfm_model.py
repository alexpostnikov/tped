import sys

sys.path.append("/home/robot/repos/tped")
from dataloader_sfm import DataloaderSfm, collate_fn
from model.model_cvae_parallel import run_lstm_on_variable_length_seqs
from utils import get_batch_is_filled_mask, setup_experiment
from dataloader_parallel import is_filled
import io
import PIL.Image

from visualize import plot_prob_big
import matplotlib.pyplot as plt

import torch
from tqdm import tqdm
import torch.nn as nn
import torch.distributions as D
from typing import List
import torch.nn.functional as F
import torch.optim as optim
from torchvision.transforms import ToTensor
STD_KOEF = 0.00
class SigmaPred(nn.Module):

    def __init__(self, lstm_hidden_dim=30, num_modes=20, num_layers=1, bidir=True, dropout_p=0.1):
        super(SigmaPred, self).__init__()

        self.num_modes = num_modes
        self.dir_number = 1
        if bidir:
            self.dir_number = 2

        self.node_hist_encoder = nn.LSTM(input_size=6,
                                         hidden_size=lstm_hidden_dim)

        self.node_hist_encoder_vel = nn.LSTM(input_size=2,
                                             hidden_size=lstm_hidden_dim,
                                             num_layers=num_layers,
                                             bidirectional=bidir,
                                             batch_first=True)

        self.node_hist_encoder_acc = nn.LSTM(input_size=2,
                                             hidden_size=lstm_hidden_dim,
                                             num_layers=num_layers,
                                             bidirectional=bidir,
                                             batch_first=True)

        self.node_hist_encoder_poses = nn.LSTM(input_size=2,
                                               hidden_size=lstm_hidden_dim,
                                               num_layers=num_layers,
                                               bidirectional=bidir,
                                               batch_first=True)

        self.node_future_encoder = nn.LSTM(input_size=6,
                                           hidden_size=lstm_hidden_dim,
                                           num_layers=num_layers,
                                           bidirectional=bidir,
                                           batch_first=True)
        self.edge_encoder = nn.LSTM(input_size=12,
                                    hidden_size=lstm_hidden_dim,
                                    num_layers=num_layers,
                                    bidirectional=bidir,
                                    batch_first=True)

        self.action = nn.Linear(6, 12)
        self.dropout_p = dropout_p
        self.gru_prep = nn.Linear(2 * lstm_hidden_dim * self.dir_number + self.action.out_features + self.num_modes,
                                  2 * lstm_hidden_dim * self.dir_number + self.action.out_features + self.num_modes)
        self.gru = nn.GRUCell(2 * lstm_hidden_dim * self.dir_number + self.action.out_features + self.num_modes,
                              num_modes * 2)

        layers = [nn.Linear(num_modes * 2, num_modes * 4),
                  nn.Dropout(self.dropout_p),
                  nn.Sigmoid(),
                  nn.Linear(num_modes * 4, 2)]
        self.h_state_to_sigmas = nn.Sequential(*layers)
        print(self.h_state_to_sigmas)

        # self.state = nn.Linear(2 * lstm_hidden_dim * self.dir_number, num_modes * 2)
        layers = [nn.Linear(2 * lstm_hidden_dim * self.dir_number, 2 * lstm_hidden_dim * self.dir_number),
                  nn.Dropout(self.dropout_p),
                  nn.Sigmoid(),
                  nn.Linear(2 * lstm_hidden_dim * self.dir_number, num_modes * 2)]
        self.state = nn.Sequential(*layers)

        # self.proj_p_to_log_pis = nn.Linear(lstm_hidden_dim * 2 * self.dir_number, num_modes)
        layers = [nn.Linear(2 * lstm_hidden_dim * self.dir_number, 2 * lstm_hidden_dim * self.dir_number),
                  nn.Dropout(self.dropout_p),
                  nn.Sigmoid(),
                  nn.Linear(2 * lstm_hidden_dim * self.dir_number, num_modes)]
        self.proj_p_to_log_pis = nn.Sequential(*layers)

        layers = [nn.Linear(3 * lstm_hidden_dim * self.dir_number, 3 * lstm_hidden_dim * self.dir_number),
                  nn.Dropout(self.dropout_p),
                  nn.Sigmoid(),
                  nn.Linear(3 * lstm_hidden_dim * self.dir_number, 2 * num_modes)]
        self.proj_to_GMM_log_pis = nn.Sequential(*layers)
        # self.proj_to_GMM_log_pis = nn.Linear(lstm_hidden_dim * 3 * self.dir_number, num_modes)

        layers = [nn.Linear(num_modes * 2, num_modes * 2), nn.Dropout(self.dropout_p), nn.Sigmoid(),
                  nn.Linear(num_modes * 2, num_modes * 2)]
        self.proj_to_GMM_mus = nn.Sequential(*layers)
        # self.proj_to_GMM_mus = nn.Linear(num_modes * 2, num_modes * 2)

        layers = [nn.Linear(num_modes * 2, num_modes * 2), nn.Dropout(self.dropout_p), nn.Sigmoid(),
                  nn.Linear(num_modes * 2, num_modes * 2)]
        self.proj_to_GMM_log_sigmas = nn.Sequential(*layers)

    def obtain_encoded_tensors(self, scene: torch.Tensor, neighbors, train):

        poses = scene[:, :8, :2]
        vel = scene[:, :8, 2:4]
        acc = scene[:, :8, 4:6]
        pav = scene[:, :8, :6]
        future = scene[:, 8:, :6]

        lstm_out_acc, hid = self.node_hist_encoder_acc(acc)  # lstm_out shape num_peds, timestamps,  2*hidden_dim
        lstm_out_vell, hid = self.node_hist_encoder_vel(vel)  # lstm_out shape num_peds, timestamps,  2*hidden_dim
        lstm_out_poses, hid = self.node_hist_encoder_poses(poses)

        lstm_out = F.dropout((lstm_out_vell + lstm_out_poses + lstm_out_acc), self.dropout_p)

        y_e = None
        if train:
            future_enc, hid = self.node_future_encoder(future)
            y_e = F.dropout(future_enc[:, -1, :], self.dropout_p)

        encoded_edges = self.encode_edge(neighbors, pav, train)
        encoded_history = torch.cat([lstm_out[:, -1, :], encoded_edges[:, -1, :]], dim=-1)

        return encoded_history, y_e

    def encode_edge(self, neighbors, node_history_st, train):

        op_applied_edge_states_list = list()
        for id, neighbors_state in enumerate(neighbors):
            op_applied_edge_states_list.append(torch.sum(neighbors_state, dim=0))
        combined_neighbors = torch.stack(op_applied_edge_states_list, dim=0).to(node_history_st.device)

        joint_history = torch.cat([combined_neighbors, node_history_st], dim=-1)

        outputs, _ = run_lstm_on_variable_length_seqs(
            self.edge_encoder,
            original_seqs=joint_history,
            lower_indices=None
        )

        outputs = F.dropout(outputs,
                            p=self.dropout_p,
                            training=train)  # [bs, max_time, enc_rnn_dim]
        return outputs

    def encoder(self, x, y_e=None, train=True):

        bs = x.shape[0]
        p_logits = F.dropout(self.proj_p_to_log_pis(x.reshape(bs, -1)), self.dropout_p)
        p_distrib = D.Normal(torch.zeros(bs, self.num_modes).to(x.device), torch.ones(bs, self.num_modes).to(x.device))
        kl = 0
        q_distrib = None

        if train:
            h = torch.cat((x.reshape(bs, -1), y_e), dim=1)
            q_logits = F.dropout(self.proj_to_GMM_log_pis(h), self.dropout_p)
            mu = q_logits[:, :self.num_modes]  # 2-d array
            logvar = q_logits[:, self.num_modes:]  # 2-d array

            var = logvar.mul(0.5).exp_()
            q_distrib = D.Normal(mu, var)

            # q_distrib = D.Normal(q_logits[:, :self.num_modes], q_logits[:, self.num_modes:])
            z = q_distrib.rsample()
            kl_separated = D.kl_divergence(q_distrib, p_distrib)
            kl = torch.clamp(torch.sum(kl_separated), min=0.001, max=1e3)
        else:
            z = p_distrib.rsample()
        return p_distrib, q_distrib, z, kl

    def decoder(self, z, encoded_history, current_state, sfm_pred, train=False):

        bs = encoded_history.shape[0]
        a_0 = F.dropout(self.action(sfm_pred[:, 0, :].reshape(bs, -1)), self.dropout_p)
        state = F.dropout(self.state(encoded_history.reshape(bs, -1)), self.dropout_p)

        current_state = current_state.unsqueeze(1)
        gauses = []
        lp = z.to(encoded_history.device)
        inp = F.dropout(torch.cat((encoded_history.reshape(bs, -1), a_0, 0 * lp), dim=-1), self.dropout_p)

        for i in range(12):

            input = inp.reshape(bs, -1)
            h_state = self.gru(input, state)
            sigma = self.h_state_to_sigmas(h_state).reshape(bs, 2)

            diag = torch.pow(sigma, 2).unsqueeze(1) * torch.eye(2).repeat(bs, 1).reshape(bs, 2, 2).to(z.device)
            #             diag += 0.001*torch.eye(2).repeat(bs,1).reshape(bs,2,2).to(z.device)
            scale_tril = torch.cholesky(diag.cpu()).to(z.device)
            gaus = D.MultivariateNormal(sfm_pred[:, i, :2], scale_tril=scale_tril)
            gauses.append(gaus)
            if (i + 1) <= 11:
                a_tt = F.dropout(self.action(sfm_pred[:, i + 1, :]), self.dropout_p)
                state = h_state
                input = torch.cat((encoded_history.reshape(bs, -1), a_tt, 0 * lp), dim=-1)
                inp = F.dropout(input, self.dropout_p)
        return gauses

    def forward(self, scene: torch.Tensor, neighbours: List, sfm_pred, train=False):

        """
        :param train:
        :param neighbours:
        :param scene: tensor of shape num_peds, history_size, data_dim
        :return: predicted poses distributions for each agent at next 12 timesteps
        """

        encoded_history, enc_future = self.obtain_encoded_tensors(scene, neighbours, train)
        p_distrib, q_distrib, z, kl = self.encoder(encoded_history, enc_future, train)
        gmm = self.decoder(z, encoded_history, scene[:, 7, :2], sfm_pred, train=train)
        if train:
            return gmm, kl
        else:
            return gmm







def training_loop(data_gen, test_gen, net, num_epochs, device, logging = True):
    net.train()
    lr = 5e-5
    optimizer = optim.Adam(net.parameters(), lr=lr)
    drop_every_epochs = 1
    drop_rate = 0.95
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, drop_every_epochs,
                                                drop_rate)  #
    writer = None
    if logging:
        writer, experiment_name, save_model_path, folder_path = setup_experiment("sigma_p", logdir="./sigma_tb/")

    for epoch in range(num_epochs):

        # #############  TRAINING  ###############
        info, net, loss = epoch_loop(device, epoch, net, optimizer, data_gen)
        if writer is not None:
            writer.add_scalar(f"loss_epoch", sum(loss["epoch_loss"])/len(loss["epoch_loss"]), epoch)
            writer.add_scalar(f"train/kl", sum(loss["epoch_loss_kl"])/len(loss["epoch_loss_kl"]), epoch)
            # writer.add_scalar(f"train/std", epoch_loss_std.detach().cpu(), epoch)
            writer.add_scalar(f"train/nll", sum(loss["epoch_loss_nll"])/len(loss["epoch_loss_nll"]), epoch)
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            writer.add_scalar(f"train/lr", lr, epoch)

        # ####################VALIDATION #####################
        with torch.no_grad():

            info, net, loss, md = epoch_loop(device, epoch, net, optimizer, test_gen, eval=True)

            if writer is not None:
                writer.add_scalar(f"test/nll", sum(loss["epoch_loss_nll"])/len(loss["epoch_loss_nll"]), epoch)

                writer.add_histogram("mahalanobis_distance", md, epoch)

            print("\t\t test loss: ", info)
            images = []
            for i in range(5):
                for batch_id, local_batch in enumerate(test_gen):
                    x, neighbours, sfm_pred = local_batch
                    x = x.to(device)
                    gt = x.clone()
                    sfm_pred = sfm_pred.to(device)
                    model.to(device)
                    model.zero_grad()

                    x = x[:, :, 2:8]
                    prediction = net(x[:, :, 0:6], neighbours, sfm_pred, train=False)
                    num_peds = x.shape[0]

                    for ped_num in range(num_peds):
                        if is_filled(x[ped_num, :8, :]):
                            if not torch.any(torch.norm(gt[ped_num, 8:, 2:4], dim=-1) == torch.tensor([0]).to(device)):
                                ax = plot_prob_big(gt[ped_num, :, 2:4], prediction, ped_num, device=device)
                                break
                    break
                buf = io.BytesIO()
                plt.savefig(buf, format='jpeg')
                buf.seek(0)
                image = PIL.Image.open(buf)
                image.save("img/temp"+str(epoch)+str(i)+".jpeg")
                images.append(ToTensor()(image))
                plt.close()
            if writer is not None:
                images = torch.cat(images, dim=1)
                writer.add_image('p_distr', images, epoch)

        scheduler.step()


def epoch_loop(device, epoch, net, optimizer, data_gen, eval=False):
    if not eval:
        pbar = tqdm(data_gen)
    else:
        pbar = data_gen
    num_processed = 0
    num_skipped = 0
    loss = {"epoch_loss": [], "epoch_loss_kl": [], "epoch_loss_nll": []}
    info = ""
    statiscits = [[]for _ in range(12)]
    for batch_id, batch in enumerate(pbar):
        x, neighbours, sfm_pred = batch
        x = x[:, :, 2:].to(device)
        bs = x.shape[0]
        sfm_pred = sfm_pred.to(device)
        gt = x.clone().to(device)
        net = net.to(device)
        net.zero_grad()
        mask, full_peds = get_batch_is_filled_mask(gt)
        num_processed += full_peds
        num_skipped += gt.size(0) - full_peds
        if full_peds == 0:
            continue

        if not eval:
            model.train()
            prediction, kl = net(x[:, :, :6], neighbours, sfm_pred, train=True)
            loss["epoch_loss_kl"].append(kl.detach().cpu().item() / bs)
            loss_stdved = 0.5 * torch.sum(torch.cat(([prediction[i].stddev for i in range(12)])))
            gt_prob = torch.stack(([prediction[i].log_prob(gt[:, 8 + i, 0:2]) for i in range(12)])).permute(1, 0)
            loss_nll = -torch.sum(torch.clamp(gt_prob * mask[:, :, 0].to(device), max=100))

            loss["epoch_loss_nll"].append(loss_nll.detach().cpu().item() / full_peds)
            kl_weight = min((4 * epoch + 1) / 100.0, 1.0)
            e_loss = 0.01 * loss_nll + kl_weight * kl + STD_KOEF * loss_stdved  # + DIST_KOEF*distance_loss
            loss["epoch_loss"].append(loss_nll.detach().cpu().item() / full_peds + kl.detach().cpu().item() / bs)
            e_loss.backward()
            info = "{epoch} ,kl: {kl:0.4f} ," \
                   " nll {nll:0.2f}".format(epoch=epoch, kl=sum(loss["epoch_loss_kl"])/len(loss["epoch_loss_kl"]),
                                            nll=sum(loss["epoch_loss_nll"])/len(loss["epoch_loss_nll"]))
            pbar.set_description(info)
            optimizer.step()

        else:

            model.eval()
            prediction = net(x[:, :, :6], neighbours, sfm_pred, train=False)
            for i in range(12):
                md = mahalanobis(prediction[i].covariance_matrix, prediction[i].mean, gt[:, 8+i, 0:2])
                md = md[torch.where(mask[:, i, 0] !=0)]
                statiscits[i].append(md)
            gt_prob = torch.stack(([prediction[i].log_prob(gt[:, 8 + i, 0:2]) for i in range(12)])).permute(1, 0)
            loss_nll = -torch.sum(gt_prob * mask[:, :, 0].to(device))
            loss["epoch_loss_nll"].append(loss_nll.detach().cpu().item() / full_peds)
            loss["epoch_loss"].append(loss_nll)
            info = " nll {nll:0.2f}".format(nll=sum(loss["epoch_loss_nll"])/len(loss["epoch_loss_nll"]))
    if eval:
        for i in range(12):
            statiscits[i] = torch.mean(torch.cat(statiscits[i]).cpu())
        statiscits = torch.stack(statiscits)
        print(statiscits)
        return info, net, loss, statiscits.reshape(12, 1)
    return info, net, loss

def mahalanobis(cov_mat, mean, gt):
    md = torch.sqrt(torch.bmm(torch.bmm((mean - gt).unsqueeze(1), torch.inverse(cov_mat)), (mean - gt).unsqueeze(2))).squeeze()
    return md

if __name__ == "__main__":
    dataset = DataloaderSfm("/home/robot/repos/tped/processed_with_forces/",
                            data_files=["eth_train.pkl"], sfm_file="eth_train_sfm.pkl")
    training_generator = torch.utils.data.DataLoader(dataset, batch_size=256, collate_fn=collate_fn)  # , num_workers=10

    test_dataset = DataloaderSfm("/home/robot/repos/tped/processed_with_forces/",
                            data_files=["eth_test.pkl"], sfm_file="eth_test_sfm.pkl")
    test_generator = torch.utils.data.DataLoader(test_dataset, batch_size=256, collate_fn=collate_fn, shuffle=True)  # , num_workers=10

    model = SigmaPred(dropout_p=0.2)

    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    training_loop(training_generator, test_generator, model, 100, dev)
