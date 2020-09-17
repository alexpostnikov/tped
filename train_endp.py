import time
from datetime import datetime
from typing import Tuple, Union
import os

import torch
import torch.optim as optim
import torch.distributions as D
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from dataloader_parallel import DatasetFromPkl, collate_fn
from model.endpoind_model import EndPointPred
from utils import compare_prediction_gt, get_batch_is_filled_mask

from google_drive_downloader import GoogleDriveDownloader as gdd

from typing import List
import tqdm
import shutil

STD_KOEF = 0.01
DROPOUT = 0.1
DIST_KOEF = 0.01

# def plot(predictions, gt):
#     num_peds = gt.shape[1]
#     for ped_num in range(num_peds):
#         if is_filled(gt[0, ped_num, :8, :]):
#             fig = plt.figure()
#             ax1 = fig.add_subplot(2, 1, 1)
#             ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
#             plot_traj(gt[0, :, :8, 2:4].detach().cpu(), ax=ax1, color="blue")
#             plot_traj(predictions[ped_num:ped_num + 1, :, :].detach().cpu(), ax2)
#             plot_traj(gt[0, ped_num:ped_num + 1, 8:, 2:4].detach().cpu(), ax2, color="black")
#             plot_traj(gt[0, ped_num:ped_num + 1, :8, 2:4].detach().cpu(), ax2, color="blue")
#             plt.show()


def setup_experiment(title: str, logdir: str = "./tb") -> Tuple[SummaryWriter, str, str, str]:
    """
    :param title: name of experiment
    :param logdir: tb logdir
    :return: writer object,  modified experiment_name, best_model path
    """
    experiment_name = "{}@{}".format(title, datetime.now().strftime("%d.%m.%Y-%H:%M:%S"))
    folder_path = os.path.join(logdir, experiment_name)
    writer = SummaryWriter(log_dir=folder_path)

    best_model_path = f"{folder_path}/{experiment_name}+best.pth"
    return writer, experiment_name, best_model_path, folder_path


def get_ml_prediction_single_timestamp(gmm: D.MixtureSameFamily):
    means = [gmm.component_distribution.mean[i][torch.argmax(gmm.mixture_distribution.logits[i]).item()]
             for i in range(gmm.mixture_distribution.logits.shape[0])]
    means = torch.cat(means).reshape(-1, 2)
    return means


def get_ml_prediction_multiple_timestamps(gmm: List[D.MixtureSameFamily]):
    means = []
    for single_gmm in gmm:
        means.append(get_ml_prediction_single_timestamp(single_gmm))
    return torch.stack(means).permute(1, 0, 2)


def get_mean_prediction_multiple_timestamps(gmm: List[D.MixtureSameFamily]):
    mean_predictions = torch.cat([gmm[i].mean for i in range(12)]).reshape(12, -1, 2).permute(1, 0, 2)
    return mean_predictions


def get_ade_fde_vel(generator: DataLoader, model: torch.nn.Module, limit: int = 1e100) -> \
        Tuple[List, List, torch.Tensor]:
    """
    :param generator: torch generator to get data
    :param model:   torch module predicting poses. input shape are [numped, history, 2]
    :param limit: limit number of processed batches. Default - unlimited
    :return: tuple of ade, fde for given generator and limit of batches
    """
    ade = []
    fde = []
    nll = []
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for batch_id, local_batch in enumerate(generator):

        if batch_id > limit:
            break

        x, neighbours = local_batch
        x = x.to(device)
        # local_batch = local_batch.to(device)
        gt = x.clone()
        mask, full_peds = get_batch_is_filled_mask(gt)
        mask = mask.to(device)
        # local_batch[0, :, 8:, :] = torch.zeros_like(local_batch[0, :, 8:, :])
        # num_peds = local_batch.shape[1]

        prediction = model(x[:, :, 2:8], neighbours)

        gt_prob = torch.cat(([prediction[i].log_prob(gt[:, 8 + i, 2:4]) for i in range(12)])).reshape(-1, 12)
        nll.append(gt_prob[mask[:, :, 0] != 0].reshape(-1, 12))
        # nll += -torch.sum(gt_prob * mask[:, :, 0])
        predictions = get_mean_prediction_multiple_timestamps(prediction)
        # if 0:
        #     predictions = []
        #     for i in range(10):
        #         # model.eval()
        #         prediction = model(local_batch[0, :, :, 2:8])
        #         p = get_mean_prediction_multiple_timestamps(prediction)
        #         predictions.append(p)
        #     model.train()
        #     predictions = torch.mean(torch.stack(predictions), dim=0)
        # predictions = get_ml_prediction_multiple_timestamps(prediction)
        if torch.any(predictions != predictions):
            print("Warn! nan pred")
            continue
        # start = time.time()
        metrics = compare_prediction_gt(predictions.detach().cpu(), gt.detach().cpu()[:, :, 2:4])
        # print(time.time() - start)
        for local_ade in metrics["ade"]:
            ade.append(local_ade)
        for local_fde in metrics["fde"]:
            fde.append(local_fde)

    return ade, fde, torch.flatten(torch.cat(nll, dim=0))


def train_endp(model: torch.nn.Module, training_generator: DataLoader,
                   test_generator: DataLoader, num_epochs: int, device: torch.device,
                   lr: Union[int, float] = 0.02, limit: Union[int, float] = 10e7, validate: bool = True, logging=False):
    """

    :param logging:
    :param validate:
    :param model: model for predicting the poses.  input shape are [numped, history, 2]
    :param training_generator:  torch training generator to get data
    :param test_generator: torch training generator to get data
    :param num_epochs: number of epochs to train
    :param device: torch device. (cpu/cuda)
    :param lr: learning rate
    :param limit: limit the number of training batches
    :return:
    """

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.05)
    drop_every_epochs = 1
    drop_rate = 0.95
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, drop_every_epochs,
                                                drop_rate)  # drop_every_epochs epochs drop by drop_rate lr
    writer = None
    dir_path = os.path.dirname(os.path.realpath(__file__))
    if logging:
        writer, experiment_name, save_model_path, folder_path = setup_experiment(model.name + str(lr) + "_hd_"
                                                                                 + str(model.lstm_hidden_dim)
                                                                                 + "_ed_" + str(model.embedding_dim)
                                                                                 + "_nl_" + str(model.num_layers))
        shutil.copyfile(f"{dir_path}/train_parallel.py", f"{folder_path}/train_parallel.py")
        shutil.copyfile(f"{dir_path}/model/model_cvae_parallel.py", f"{folder_path}/model_cvae_parallel.py")
    prev_epoch_loss = 0
    min_ade = 1e100

    for epoch in range(0, num_epochs):
        model.train()
        epoch_loss = 0
        epoch_loss_nll = 0
        epoch_loss_kl = 0
        epoch_loss_std = 0
        start = time.time()
        num_skipped = 0
        num_processed = 0

        if writer is not None:
            visualize_model_weights(epoch, model, writer)

        batch_id = 0
        pbar = tqdm.tqdm(training_generator)
        for batch_id, local_batch in enumerate(pbar):
            if batch_id - num_skipped > limit:
                # skip to next epoch
                break

            # angle = torch.rand(1) * math.pi
            # rot = torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
            # rotrot = torch.zeros(4, 4)
            # rotrot[:2, :2] = rot.clone()
            # rotrot[2:, 2:] = rot.clone()
            # local_batch[:, :, :, 2:6] = local_batch[:, :, :, 2:6] @ rotrot
            x, neighbours = local_batch
            x = x.to(device)
            gt = x.clone()
            model = model.to(device)
            model.zero_grad()
            mask, full_peds = get_batch_is_filled_mask(gt)
            num_processed += full_peds

            num_skipped += gt.size(0) - full_peds
            if full_peds == 0:
                continue
            mask = mask.to(device)

            x = x[:, :, 2:8]
            if torch.sum(mask) == 0.0:
                num_skipped += 1
                continue
            pred_d, kl = model(x[:, :, 0:6], neighbours, train=True)
            # gt_prob = torch.cat(([prediction[i].log_prob(gt[:, 8 + i, 2:4]) for i in range(12)])).reshape(-1, 12)
            # loss_nll = -torch.sum(gt_prob * mask[:, :, 0])
            # epoch_loss_nll += loss_nll
            epoch_loss_kl += kl

            # means = get_mean_prediction_multiple_timestamps(prediction)
            dest_loss = torch.dist(pred_d * mask[:, -1, :], gt[:, -1, 2:4] * mask[:, -1, :])
            # distance_loss = torch.dist(means* mask, gt[:, 8:, 2:4] * mask)
            # loss_stdved = 0.5 * torch.sum(torch.cat(([prediction[i].stddev for i in range(12)])))
            # epoch_loss_std += loss_stdved

            info = "kl: {kl:0.4f} ," \
                   " l2_d {l2_d:0.2f}".format(kl=epoch_loss_kl.detach().cpu().item() / num_processed,
                                              l2_d=dest_loss / full_peds)
            pbar.set_description(info)
            kl_weight = min((5*(epoch+1))/100.0, 1.0)
            loss = dest_loss + kl_weight * kl
            # loss = 0.01 * loss_nll + kl_weight * kl + STD_KOEF * loss_stdved + DIST_KOEF*distance_loss

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        epoch_loss = epoch_loss / num_processed
        # epoch_loss_kl = epoch_loss_kl / num_processed
        # epoch_loss_nll = epoch_loss_nll / num_processed
        # epoch_loss_std = epoch_loss_std / num_processed
        # print("\tepoch {epoch} loss {el:0.4f}, time taken {t:0.2f},"
        #       " delta {delta:0.3f}".format(epoch=epoch, el=epoch_loss,
        #                                    t=time.time() - start,
        #                                    delta=-prev_epoch_loss + epoch_loss))

        if writer is not None:
            writer.add_scalar(f"loss_epoch", epoch_loss, epoch)
            writer.add_scalar(f"train/kl", epoch_loss_kl.detach().cpu(), epoch)
            writer.add_scalar(f"train/std", epoch_loss_std.detach().cpu(), epoch)
            writer.add_scalar(f"train/nll", epoch_loss_nll.cpu(), epoch)
            for param_group in optimizer.param_groups:
                lr = param_group['lr']
            writer.add_scalar(f"train/lr", lr, epoch)
        prev_epoch_loss = epoch_loss

        # ## VALIDATION ###
        num_processed = 0
        num_skipped = 0
        dest_loss_test_set = 0
        if validate and (epoch % 1 == 0):
            model.eval()
            start = time.time()
            with torch.no_grad():
                for batch_id, local_batch in enumerate(test_generator):
                    x, neighbours = local_batch
                    x = x.to(device)
                    gt = x.clone()
                    model = model.to(device)
                    model.zero_grad()
                    mask, full_peds = get_batch_is_filled_mask(gt)
                    num_processed += full_peds

                    num_skipped += gt.size(0) - full_peds
                    if full_peds == 0:
                        continue
                    mask = mask.to(device)

                    x = x[:, :, 2:8]
                    if torch.sum(mask) == 0.0:
                        num_skipped += 1
                        continue
                    pred_d = model(x[:, :, 0:6], neighbours, train=False)

                    dest_loss = torch.dist(pred_d * mask[:, -1, :], gt[:, -1, 2:4] * mask[:, -1, :])
                    dest_loss_test_set += dest_loss
            print ("\t\t dest_loss_test_set: ", dest_loss_test_set / num_processed)
        scheduler.step()
    if writer is not None:
        torch.save(model.state_dict(), save_model_path)


def visualize_model_weights(epoch, model, writer):
    weights = torch.cat([i.data.flatten() for i in model.node_hist_encoder.all_weights[0]]).detach().cpu()
    writer.add_histogram(f"model/node_hist_encoder", weights, epoch)
    weights = torch.cat(
        [i.data.flatten() for i in model.node_hist_encoder_vel.all_weights[0]]).detach().cpu()
    writer.add_histogram(f"model/node_hist_encoder_vel", weights, epoch)
    weights = torch.cat(
        [i.data.flatten() for i in model.node_hist_encoder_acc.all_weights[0]]).detach().cpu()
    writer.add_histogram(f"model/node_hist_encoder_acc", weights, epoch)
    weights = torch.cat(
        [i.data.flatten() for i in model.node_hist_encoder_poses.all_weights[0]]).detach().cpu()
    writer.add_histogram(f"model/node_hist_encoder_poses", weights, epoch)
    weights = torch.cat(
        [i.data.flatten() for i in model.edge_encoder.all_weights[0]]).detach().cpu()
    writer.add_histogram(f"model/edge_encoder", weights, epoch)
    weights = torch.cat(
        [i.data.flatten() for i in model.node_future_encoder.all_weights[0]]).detach().cpu()
    writer.add_histogram(f"model/node_future_encoder", weights, epoch)
    # ### GRU ####
    weights = torch.cat([model.gru.weight_hh.data.flatten(), model.gru.weight_ih.data.flatten()])
    writer.add_histogram(f"model/gru", weights, epoch)
    # ########## LINEAR ##################
    # weights = model.action.weight.data.flatten().detach().cpu()
    # writer.add_histogram(f"model/action", weights, epoch)
    # weights = model.state.weight.data.flatten().detach().cpu()
    # writer.add_histogram(f"model/state", weights, epoch)
    # weights = model.proj_p_to_log_pis.weight.data.flatten().detach().cpu()
    # writer.add_histogram(f"model/proj_p_to_log_pis", weights, epoch)
    # weights = model.proj_to_GMM_log_pis.weight.data.flatten().detach().cpu()
    # writer.add_histogram(f"model/proj_to_GMM_log_pis", weights, epoch)
    # weights = model.proj_to_GMM_mus.weight.data.flatten().detach().cpu()
    # writer.add_histogram(f"model/proj_to_GMM_mus", weights, epoch)
    # weights = model.proj_to_GMM_log_sigmas.weight.data.flatten().detach().cpu()
    # writer.add_histogram(f"model/proj_to_GMM_log_sigmas", weights, epoch)
    # weights = model.proj_to_GMM_corrs.weight.data.flatten().detach().cpu()
    # writer.add_histogram(f"model/proj_to_GMM_corrs", weights, epoch)


if __name__ == "__main__":
    print(os.path.isdir("./processed_with_forces"))
    if not os.path.isdir("./processed_with_forces"):
        gdd.download_file_from_google_drive(file_id='1K0WPpglh2fEuLYMoDNuuEzftzoYTG1p6',
                                            dest_path='./processed_with_forces.zip',
                                            unzip=True)

        gdd.download_file_from_google_drive(file_id='11U5vcq4gyc3Fr9DuFmxbYTjENtE2a-LY',
                                            dest_path='./preprocessed.zip',
                                            unzip=True)

    training_set = DatasetFromPkl("./processed_with_forces/", data_files=["eth_train.pkl"])
    training_gen = DataLoader(training_set, batch_size=256, shuffle=True,
                              collate_fn=collate_fn)  # , num_workers=4

    test_set = DatasetFromPkl("./processed_with_forces/", data_files=["eth_test.pkl"])
    test_gen = DataLoader(test_set, batch_size=32, shuffle=True, collate_fn=collate_fn)  # , num_workers=10
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("starting training on ", dev)

    net = EndPointPred(lstm_hidden_dim=32, num_layers=1, bidir=True, dropout_p=DROPOUT, num_modes=40).to(dev)

    net.name += "_STD"+str(STD_KOEF)+"_D"+str(DROPOUT)+"_DIST"+str(DIST_KOEF)+"_"
    print("model name: ", net.name)
    train_endp(net, training_gen, test_gen, num_epochs=300, device=dev, lr=0.0005,
                   limit=1e100, logging=False)
