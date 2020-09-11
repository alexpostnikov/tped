import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributions as D
from torch.autograd import Variable
from typing import List
import torch.nn.utils.rnn as rnn


def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot


def run_lstm_on_variable_length_seqs(lstm_module, original_seqs, lower_indices=None, upper_indices=None,
                                     total_length=None):
    bs, tf = original_seqs.shape[:2]
    if lower_indices is None:
        lower_indices = torch.zeros(bs, dtype=torch.int)
    if upper_indices is None:
        upper_indices = torch.ones(bs, dtype=torch.int) * (tf - 1)
    if total_length is None:
        total_length = max(upper_indices) + 1
    # This is done so that we can just pass in self.prediction_timesteps
    # (which we want to INCLUDE, so this will exclude the next timestep).
    inclusive_break_indices = upper_indices + 1

    pad_list = list()
    for i, seq_len in enumerate(inclusive_break_indices):
        pad_list.append(original_seqs[i, lower_indices[i]:seq_len])

    packed_seqs = rnn.pack_sequence(pad_list, enforce_sorted=False)
    packed_output, (h_n, c_n) = lstm_module(packed_seqs)
    output, _ = rnn.pad_packed_sequence(packed_output,
                                        batch_first=True,
                                        total_length=total_length)

    return output, (h_n, c_n)


def sample(p_distrib: D.Categorical):
    z = p_distrib.sample()
    return z


class EndPointPred(nn.Module):
    """
    model that actually predicts delta movements (but output last timestamp + predicted deltas),
    with encoding person history and neighbors relative positions.
    """

    def __init__(self, lstm_hidden_dim, num_layers=1, bidir=True, dropout_p=0.5, num_modes=20):
        super(EndPointPred, self).__init__()
        self.name = "CvaeFuture_parallel"
        self.num_modes = num_modes
        self.embedding_dim = 0
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_layers = num_layers
        self.dir_number = 1
        if bidir:
            self.dir_number = 2
        self.node_hist_encoder = nn.LSTM(input_size=6,
                                         hidden_size=lstm_hidden_dim,
                                         num_layers=num_layers,
                                         bidirectional=bidir,
                                         batch_first=True)

        self.bn1 = nn.BatchNorm1d(8)
        self.bn2 = nn.BatchNorm1d(8)
        self.bn3 = nn.BatchNorm1d(num_modes * 2)
        self.bn4 = nn.BatchNorm1d(num_modes * 2)

        self.ln1 = nn.LayerNorm([8, lstm_hidden_dim])
        self.ln2 = nn.LayerNorm([8, lstm_hidden_dim])
        self.ln3 = nn.LayerNorm([num_modes * 2])
        self.ln4 = nn.LayerNorm([num_modes * 2])

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

        self.edge_encoder = nn.LSTM(input_size=12,
                                    hidden_size=lstm_hidden_dim,
                                    num_layers=num_layers,
                                    bidirectional=bidir,
                                    batch_first=True)

        self.node_future_encoder = nn.LSTM(input_size=6,
                                           hidden_size=lstm_hidden_dim,
                                           num_layers=num_layers,
                                           bidirectional=bidir,
                                           batch_first=True)
        self.dest_encoder_out_shape = self.dir_number * 12
        self.dest_encoder = nn.LSTM(input_size=6,
                                           hidden_size=self.dest_encoder_out_shape // self.dir_number,
                                           num_layers=num_layers,
                                           bidirectional=bidir,
                                           batch_first=True)
        self.dropout_p = dropout_p
        layers = [nn.Linear(2 * lstm_hidden_dim * self.dir_number + (self.num_modes // 2), 64),
                  nn.Dropout(self.dropout_p),
                  nn.Sigmoid(),
                  nn.Linear(64, 2)]
        self.dest_decoder = nn.Sequential(*layers)

        self.action = nn.Linear(2, 12)

        self.gru_prep = nn.Linear(2 * lstm_hidden_dim * self.dir_number + self.action.out_features + self.num_modes,
                                  2*lstm_hidden_dim * self.dir_number + self.action.out_features + self.num_modes)
        self.gru = nn.GRUCell(2 * lstm_hidden_dim * self.dir_number + self.action.out_features + self.num_modes,
                              num_modes * 2)

        # self.state = nn.Linear(2 * lstm_hidden_dim * self.dir_number, num_modes * 2)
        layers = [nn.Linear(2 * lstm_hidden_dim * self.dir_number, 2 * lstm_hidden_dim * self.dir_number),
                  nn.Dropout(self.dropout_p),
                  nn.Sigmoid(),
                  nn.Linear(2 * lstm_hidden_dim * self.dir_number, num_modes * 2)]
        self.state = nn.Sequential(*layers)

        # self.proj_p_to_log_pis = nn.Linear(lstm_hidden_dim * 2 * self.dir_number, num_modes)
        layers = [nn.Linear(1 * lstm_hidden_dim * self.dir_number, 1 * lstm_hidden_dim * self.dir_number),
                  nn.Dropout(self.dropout_p),
                  nn.Sigmoid(),
                  nn.Linear(1 * lstm_hidden_dim * self.dir_number, num_modes)]
        self.proj_p_to_log_pis = nn.Sequential(*layers)

        layers = [nn.Linear(2 * lstm_hidden_dim * self.dir_number + self.dest_encoder_out_shape,
                            2 * lstm_hidden_dim * self.dir_number + self.dest_encoder_out_shape),
                  nn.Dropout(self.dropout_p),
                  nn.Sigmoid(),
                  nn.Linear(2 * lstm_hidden_dim * self.dir_number + self.dest_encoder_out_shape, num_modes)]
        self.proj_to_GMM_log_pis = nn.Sequential(*layers)
        # self.proj_to_GMM_log_pis = nn.Linear(lstm_hidden_dim * 3 * self.dir_number, num_modes)

        layers = [nn.Linear(num_modes * 2, num_modes * 2), nn.Dropout(self.dropout_p), nn.Sigmoid(),
                  nn.Linear(num_modes * 2, num_modes * 2)]
        self.proj_to_GMM_mus = nn.Sequential(*layers)
        # self.proj_to_GMM_mus = nn.Linear(num_modes * 2, num_modes * 2)

        layers = [nn.Linear(num_modes * 2, num_modes * 2), nn.Dropout(self.dropout_p), nn.Sigmoid(),
                  nn.Linear(num_modes * 2, num_modes * 2)]
        self.proj_to_GMM_log_sigmas = nn.Sequential(*layers)
        # self.proj_to_GMM_log_sigmas = nn.Linear(num_modes * 2, num_modes * 2)

        self.proj_to_GMM_corrs = nn.Linear(num_modes * 2, num_modes)
        self.proj_z_to_log_pi = nn.Linear(num_modes, num_modes)


    def project_to_gmm_params(self, tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
        """
        Projects tensor to parameters of a GMM with N components and D dimensions.

        :param tensor: Input tensor.
        :return: tuple(log_pis, mus, log_sigmas, corrs)
            WHERE
            - log_pis: Weight (logarithm) of each GMM component. [N]
            - mus: Mean of each GMM component. [N, D]
            - log_sigmas: Standard Deviation (logarithm) of each GMM component. [N, D]
            - corrs: Correlation between the GMM components. [N]
        """
        # log_pis =  F.dropout(self.proj_to_GMM_log_pis(tensor), self.dropout_p)
        mus = F.dropout(self.proj_to_GMM_mus(tensor), self.dropout_p)
        log_sigmas = F.dropout(self.proj_to_GMM_log_sigmas(tensor), self.dropout_p)
        corrs = F.dropout(torch.tanh(self.proj_to_GMM_corrs(tensor)), self.dropout_p)
        return None, mus, log_sigmas, corrs

    def sample_q(self):
        pass

    def sample_p(self):
        pass

    def obtain_encoded_tensors(self, scene: torch.Tensor, neighbors, train):
        # bs = scene.shape[0]
        poses = scene[:, :8, :2]
        # pv = scene[:, :8, 2:6]
        vel = scene[:, :8, 2:4]
        acc = scene[:, :8, 4:6]
        pav = scene[:, :8, :6]
        future = scene[:, 8:, :6]
        dest = future = scene[:, -1:, :6]

        # lstm_out, hid = self.node_hist_encoder(pav)  # lstm_out shape num_peds, timestamps , 2*hidden_dim
        lstm_out_acc, hid = self.node_hist_encoder_acc(acc)  # lstm_out shape num_peds, timestamps,  2*hidden_dim
        lstm_out_vell, hid = self.node_hist_encoder_vel(vel)  # lstm_out shape num_peds, timestamps,  2*hidden_dim
        lstm_out_poses, hid = self.node_hist_encoder_poses(poses)

        # lstm_out = self.bn1(lstm_out_vell + lstm_out_poses + lstm_out_acc)
        # lstm_out = self.ln1(F.dropout(self.ln1(lstm_out_vell + lstm_out_poses + lstm_out_acc), self.dropout_p))
        lstm_out = F.dropout((lstm_out_vell + lstm_out_poses + lstm_out_acc), self.dropout_p)
        # lstm_out = F.dropout((lstm_out_poses), self.dropout_p)

        y_e = None
        if train:
            future_enc, hid = self.node_future_encoder(future)
            dest_enc, hid = self.dest_encoder(dest)
            # y_e = F.dropout(torch.cat((future_enc[:, -1, :], dest_enc[:, -1, :]), dim=-1), self.dropout_p)
            y_e = F.dropout(dest_enc[:, -1, :], self.dropout_p)

        # np, data_dim = current_pose.shape
        # stacked = current_pose.flatten().repeat(np).reshape(np, np * data_dim)
        # deltas = (stacked - current_pose.repeat(1, np)).reshape(np, np, data_dim)  # np, np, data_dim
        #
        # distruction, _ = self.edge_encoder(deltas)
        # encoded_history = torch.cat((lstm_out[:, -1:, :], distruction[:, -1:, :]), dim=1)
        # encoded_edges = self.bn2(self.encode_edge(neighbors, pav, train))
        # encoded_edges = self.ln2(self.encode_edge(neighbors, pav, train))
        encoded_edges = self.encode_edge(neighbors, pav, train)
        encoded_history = torch.cat([lstm_out[:, -1, :], encoded_edges[:, -1, :]], dim=-1)


        return encoded_history, y_e , None

    def encode_edge(self, neighbors, node_history_st, train):

        # edge_states_list = []
        # for i, neighbor_states in enumerate(neighbors):  # Get neighbors for timestep in batch
        #     if len(neighbor_states) == 0:  # There are no neighbors for edge type
        #         pass
        #         # neighbor_state_length = int(
        #         #     np.sum([len(entity_dims) for entity_dims in self.state[edge_type[1]].values()])
        #         # )
        #         # edge_states_list.append(torch.zeros((1, max_hl + 1, neighbor_state_length), device=self.device))
        #     else:
        #         edge_states_list.append(torch.stack(neighbor_states, dim=0).to(self.device))

        # if self.hyperparams['edge_state_combine_method'] == 'sum':
        # Used in Structural-RNN to combine edges as well.
        op_applied_edge_states_list = list()
        for id, neighbors_state in enumerate(neighbors):
            # if len(neighbors_state) > 0:
            #     clothest_id = torch.argmin(
            #         torch.sum(((neighbors_state.cpu()[:, :, 0:2] - node_history_st[id, :, 0:2].cpu()) ** 2)[:, :, 0],
            #                   dim=-1))
            #     op_applied_edge_states_list.append(neighbors_state[clothest_id].to(node_history_st.device) - node_history_st[id])
            # else:
            #     op_applied_edge_states_list.append(torch.zeros(8, 6).to(node_history_st.device))
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
        # q_distrib -> onehot?
        # p_distrib -> onehot?
        # kl dist
        # sample z
        bs = x.shape[0]
        p_distrib = D.Normal(torch.zeros(bs, self.num_modes//2).to(x.device), torch.ones(bs, self.num_modes//2).to(x.device))
        # z = torch.Tensor(bs, 25).to(x.device)
        # z.normal_(0, 25).to(x.device)
        kl = 0
        q_distrib = None

        if train:
            h = torch.cat((x.reshape(bs, -1), y_e), dim=1)
            q_logits = F.dropout(self.proj_to_GMM_log_pis(h), self.dropout_p)
            mu = q_logits[:, 0:self.num_modes//2]  # 2-d array
            logvar = q_logits[:, self.num_modes//2:]  # 2-d array

            var = logvar.mul(0.5).exp_()
            q_distrib = D.Normal(mu, var)
            eps = torch.DoubleTensor(var.size()).normal_()
            # eps = eps.to(x.device)
            # z = eps.mul(var).add_(mu)
            z = q_distrib.rsample()
            # z = q_logits
            kl_separated = D.kl_divergence(q_distrib, p_distrib)
            # kl_lower_bounded = torch.clamp(, min=0.1, max=1e3)
            kl = torch.clamp(torch.sum(kl_separated), min=1e-6, max=1e10)  # - torch.sum(q_distrib.entropy())
        else:
            z = p_distrib.rsample()
            # z = p_distrib.rsample(2)
            # z = p_logits
        return p_distrib, q_distrib, z, kl

    def decoder(self, z, encoded_history, current_state, train=False):
        pred_dest = self.dest_decoder(torch.cat((z, encoded_history), dim=-1))
        return pred_dest


    def loss(self, gmm, kl):
        # ELBO = log_likelihood - self.kl_weight * kl + 1. * mutual_inf_p
        pass

    def forward(self, scene: torch.Tensor, neighbours: List, train=False):
        """
        :param train:
        :param neighbours:
        :param scene: tensor of shape num_peds, history_size, data_dim
        :return: predicted poses distributions for each agent at next 12 timesteps
        """

        encoded_history, enc_future, pred_d = self.obtain_encoded_tensors(scene, neighbours, train)
        p_distrib, q_distrib, z, kl = self.encoder(encoded_history, enc_future, train)
        pred_d = self.decoder(z, encoded_history, scene[:, 7, :2], train=train)
        if train:
            return pred_d, kl
        else:
            return pred_d
