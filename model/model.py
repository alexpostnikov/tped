import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.multivariate_normal import MultivariateNormal
import torch.distributions as D



class LSTMTagger(nn.Module):

    def __init__(self, hidden_dim, input_dim, tagset_size, num_pred=12):
        super(LSTMTagger, self).__init__()
        self.num_pred = num_pred
        self.hidden_dim = hidden_dim

        #         self.pose_embeddings = nn.Linear(input_dim, embedding_dim)

        self.node_future_encoder = nn.LSTM(input_size=input_dim,
                                           hidden_size=hidden_dim,
                                           bidirectional=True,
                                           batch_first=True)

        self.edge_encoder = nn.LSTM(input_size=input_dim * 2,
                                    hidden_size=hidden_dim,
                                    bidirectional=True,
                                    batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(2 * hidden_dim, tagset_size)

    def forward(self, self_pose, others_pose):
        lstm_out, _ = self.node_future_encoder(self_pose)
        for person in range(others_pose.shape[1]):
            other_pose = others_pose[:, person, :, :]
            concated = torch.cat((self_pose, other_pose), dim=-1)
            distr, _ = self.edge_encoder(concated)
        #             distr , edge_enc_hiden = self.edge_encoder(concated, edge_enc_hiden)

        tag_space = self.hidden2tag(lstm_out + distr)

        return tag_space[:, -1:, :]


class LstmHid(nn.Module):

    def __init__(self, hidden_dim, input_dim, tagset_size, num_pred=12):
        super(LstmHid, self).__init__()
        self.num_pred = num_pred
        self.hidden_dim = hidden_dim
        self.node_future_encoder = nn.LSTM(input_size=input_dim,
                                           hidden_size=hidden_dim,
                                           bidirectional=True,
                                           batch_first=True)

        self.edge_encoder = nn.LSTM(input_size=input_dim * 2,
                                    hidden_size=hidden_dim,
                                    bidirectional=True,
                                    batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.edge_hidden = torch.zeros(2, 1, hidden_dim)
        self.edge_state = torch.zeros(2, 1, hidden_dim)
        self.hidden2tag = nn.Linear(2 * hidden_dim, tagset_size)

    def forward(self, self_pose, others_pose):
        lstm_out, _ = self.node_future_encoder(self_pose)
        self.edge_hidden = torch.zeros(2, 1, self.hidden_dim).to(self_pose.device)
        self.edge_state = torch.zeros(2, 1, self.hidden_dim).to(self_pose.device)
        for person in range(others_pose.shape[1]):
            other_pose = others_pose[:, person, :, :]
            concated = torch.cat((self_pose, other_pose), dim=-1)
            distr, (self.edge_hidden, self.edge_state) = self.edge_encoder(concated,
                                                                           (self.edge_hidden, self.edge_state))

        tag_space = self.hidden2tag(lstm_out + distr)

        return tag_space[:, -1:, :]


class LstmSimple(nn.Module):

    def __init__(self, hidden_dim, input_dim, tagset_size, num_pred=12):
        super(LstmSimple, self).__init__()
        self.num_pred = num_pred
        self.name = "LSTM_simple"
        self.embedding_dim = 0
        self.num_layers = 0
        self.lstm_hidden_dim = hidden_dim
        self.node_future_encoder = nn.LSTM(input_size=input_dim,
                                           hidden_size=hidden_dim,
                                           bidirectional=True,
                                           batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(2 * hidden_dim, tagset_size)

    def forward(self, self_pose):
        lstm_out, _ = self.node_future_encoder(self_pose)

        tag_space = self.hidden2tag(lstm_out)

        return tag_space[:, -1:, :]


class LSTM_single(nn.Module):

    def __init__(self, hidden_dim, input_dim, tagset_size, num_pred=12):
        super(LSTM_single, self).__init__()
        self.num_pred = num_pred
        self.name = "LSTM_single"
        self.embedding_dim = 0
        self.num_layers = 0
        self.lstm_hidden_dim = hidden_dim
        self.node_future_encoder = nn.LSTM(input_size=input_dim,
                                           hidden_size=hidden_dim,
                                           bidirectional=True,
                                           batch_first=True)

        self.edge_encoder = nn.LSTM(input_size=input_dim,
                                    hidden_size=hidden_dim,
                                    bidirectional=True,
                                    batch_first=True)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(2 * hidden_dim, tagset_size)

    def forward(self, scene):
        # scene shape = num_peds, timestamps, data_dim
        lstm_out, _ = self.node_future_encoder(scene)  # lstm_out shape num_peds, timestamps ,  2*hidden_dim
        distr, _ = self.edge_encoder(scene)  # shape num_peds, timestamps ,  2*hidden_dim
        tag_space = self.hidden2tag(lstm_out + distr)

        return tag_space[:, -1:, :]


class LSTM_single_with_emb(nn.Module):

    def __init__(self, hidden_dim, input_dim, tagset_size, num_pred=12):
        super(LSTM_single_with_emb, self).__init__()
        self.num_pred = num_pred
        self.input_dim = input_dim
        self.inp_emb = nn.Linear(in_features=2, out_features=input_dim)
        self.current_inp_emb = nn.Linear(in_features=2, out_features=input_dim)
        self.node_future_encoder = nn.LSTM(input_size=input_dim,
                                           hidden_size=hidden_dim,
                                           num_layers=3,
                                           bidirectional=True,
                                           batch_first=True,
                                           dropout=0.5)

        self.edge_encoder = nn.LSTM(input_size=input_dim,
                                    hidden_size=hidden_dim,
                                    num_layers=3,
                                    bidirectional=True,
                                    batch_first=True,
                                    dropout=0.5)
        self.hidden_dim = hidden_dim
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(18 * hidden_dim, tagset_size)

    def forward(self, scene):
        inp = scene
        embedded = F.relu(self.inp_emb(inp).reshape(scene.shape[0], -1, self.input_dim))
        # scene shape = num_peds, timestamps, data_dim
        lstm_out, _ = self.node_future_encoder(embedded)  # lstm_out shape num_peds, timestamps ,  2*hidden_dim
        current = scene[:, -1, :]
        current_emb = F.relu(self.current_inp_emb(current))
        distr, _ = self.edge_encoder(current_emb.unsqueeze(1))  # shape num_peds, timestamps ,  2*hidden_dim
        catted = torch.cat((lstm_out, distr), dim=1).reshape(-1, 18 * self.hidden_dim)
        tag_space = self.hidden2tag(catted).unsqueeze(1)

        return tag_space


class LSTM_delta(nn.Module):

    def __init__(self, lstm_hidden_dim, embedding_dim, target_size, num_layers=1):
        super(LSTM_delta, self).__init__()

        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.inp_emb = nn.Linear(in_features=2, out_features=embedding_dim)
        self.current_inp_emb = nn.Linear(in_features=2, out_features=embedding_dim)
        self.node_hist_encoder = nn.LSTM(input_size=embedding_dim,
                                         hidden_size=lstm_hidden_dim,
                                         num_layers=num_layers,
                                         bidirectional=True,
                                         batch_first=True,
                                         dropout=0.5)

        self.edge_encoder = nn.LSTM(input_size=embedding_dim,
                                    hidden_size=lstm_hidden_dim,
                                    num_layers=num_layers,
                                    bidirectional=True,
                                    batch_first=True,
                                    dropout=0.5)

        # The linear layer that maps from hidden state space to tag space
        self.hidden2pose = nn.Linear(18 * lstm_hidden_dim, target_size)

    def forward(self, scene: torch.Tensor) -> torch.Tensor:
        """
        :param scene: tensor of shape num_peds, history_size, data_dim
        :return: predicted poses for each agent at next timestep
        """

        inp = scene
        embedded = F.relu(self.inp_emb(inp))
        embedded = embedded.reshape(scene.shape[0], -1, self.embedding_dim)
        # scene shape = num_peds, timestamps, data_dim
        lstm_out, _ = self.node_hist_encoder(embedded)  # lstm_out shape num_peds, timestamps ,  2*hidden_dim
        current = scene[:, -1, :]
        current_emb = F.relu(self.current_inp_emb(current))
        distr, _ = self.edge_encoder(current_emb.unsqueeze(1))  # shape num_peds, timestamps ,  2*hidden_dim
        catted = torch.cat((lstm_out, distr), dim=1).reshape(-1, 18 * self.lstm_hidden_dim)
        tag_space = (scene.clone()[:, -1, :] + self.hidden2pose(catted)).unsqueeze(1)

        return tag_space


class LSTM_enc_delta(nn.Module):

    def __init__(self, lstm_hidden_dim, embedding_dim, target_size, num_layers=1):
        super(LSTM_enc_delta, self).__init__()
        self.name = "LSTM_enc_delta"
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_layers = num_layers
        self.inp_emb = nn.Linear(in_features=2, out_features=embedding_dim)
        self.current_inp_emb = nn.Linear(in_features=2, out_features=embedding_dim)
        self.node_hist_encoder = nn.LSTM(input_size=embedding_dim,
                                         hidden_size=lstm_hidden_dim,
                                         num_layers=num_layers,
                                         bidirectional=True,
                                         batch_first=True,
                                         dropout=0.5)

        self.edge_encoder = nn.LSTM(input_size=embedding_dim,
                                    hidden_size=lstm_hidden_dim,
                                    num_layers=num_layers,
                                    bidirectional=True,
                                    batch_first=True,
                                    dropout=0.5)

        self.decoder = nn.LSTM(input_size=2 * lstm_hidden_dim,
                               hidden_size=embedding_dim,
                               num_layers=num_layers,
                               bidirectional=True,
                               batch_first=True,
                               dropout=0.5)

        # The linear layer that maps from hidden state space to tag space

        self.hidden2pose = nn.Linear(9 * 2 * embedding_dim, target_size)

    def forward(self, scene: torch.Tensor) -> torch.Tensor:
        """
        :param scene: tensor of shape num_peds, history_size, data_dim
        :return: predicted poses for each agent at next timestep
        """
        bs = scene.shape[0]
        inp = scene
        embedded = F.relu(self.inp_emb(inp))
        embedded = embedded.reshape(scene.shape[0], -1, self.embedding_dim)
        # scene shape = num_peds, timestamps, data_dim
        lstm_out, _ = self.node_hist_encoder(embedded)  # lstm_out shape num_peds, timestamps ,  2*hidden_dim
        current = scene[:, -1, :]
        current_emb = F.relu(self.current_inp_emb(current))
        distr, _ = self.edge_encoder(current_emb.unsqueeze(1))  # shape num_peds, timestamps ,  2*hidden_dim
        catted = torch.cat((lstm_out, distr), dim=1)
        decoded, _ = self.decoder(catted)
        dec_reshaped = decoded.reshape(bs, -1)
        tag_space = (scene.clone()[:, -1, :] + self.hidden2pose(dec_reshaped)).unsqueeze(1)

        return tag_space


class LSTM_enc_delta_wo_emb(nn.Module):

    def __init__(self, lstm_hidden_dim, target_size, num_layers=1, embedding_dim=10):
        super(LSTM_enc_delta_wo_emb, self).__init__()
        self.name = "LSTM_enc_delta_wo_emb"
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_layers = num_layers
        # self.inp_emb = nn.Linear(in_features=2, out_features=embedding_dim)
        # self.current_inp_emb = nn.Linear(in_features=2, out_features=embedding_dim)
        self.node_hist_encoder = nn.LSTM(input_size=2,
                                         hidden_size=lstm_hidden_dim,
                                         num_layers=num_layers,
                                         bidirectional=True,
                                         batch_first=True,
                                         dropout=0.5)

        self.edge_encoder = nn.LSTM(input_size=2,
                                    hidden_size=lstm_hidden_dim,
                                    num_layers=num_layers,
                                    bidirectional=True,
                                    batch_first=True,
                                    dropout=0.5)

        self.decoder = nn.LSTM(input_size=2 * lstm_hidden_dim,
                               hidden_size=embedding_dim,
                               num_layers=num_layers,
                               bidirectional=True,
                               batch_first=True,
                               dropout=0.5)

        # The linear layer that maps from hidden state space to tag space

        self.hidden2pose = nn.Linear(9 * 2 * embedding_dim, target_size)

    def forward(self, scene: torch.Tensor) -> torch.Tensor:
        """
        :param scene: tensor of shape num_peds, history_size, data_dim
        :return: predicted poses for each agent at next timestep
        """
        bs = scene.shape[0]
        inp = scene
        # embedded = embedded.reshape(scene.shape[0], -1, self.embedding_dim)
        # scene shape = num_peds, timestamps, data_dim
        lstm_out, _ = self.node_hist_encoder(inp)  # lstm_out shape num_peds, timestamps ,  2*hidden_dim
        current = scene[:, -1, :]
        # current_emb = F.relu(self.current_inp_emb(current))
        distr, _ = self.edge_encoder(current.unsqueeze(1))  # shape num_peds, timestamps ,  2*hidden_dim
        catted = torch.cat((lstm_out, distr), dim=1)
        decoded, _ = self.decoder(catted)
        dec_reshaped = decoded.reshape(bs, -1)
        tag_space = (scene.clone()[:, -1, :] + self.hidden2pose(dec_reshaped)).unsqueeze(1)

        return tag_space


class LstmEncDeltaStacked(nn.Module):
    """
    model that actually predicts delta movements (output last timestamp + predicted delta),
    with encoding person history and neighbors relative positions.
    """

    def __init__(self, lstm_hidden_dim, target_size, num_layers=1, embedding_dim=10, bidir=True):
        super(LstmEncDeltaStacked, self).__init__()
        self.name = "LstmEncDeltaStacked"
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_layers = num_layers
        self.dir_number = 1
        if bidir:
            self.dir_number = 2
        self.node_hist_encoder = nn.LSTM(input_size=2,
                                         hidden_size=lstm_hidden_dim,
                                         num_layers=num_layers,
                                         bidirectional=bidir,
                                         batch_first=True,
                                         dropout=0.5)

        self.edge_encoder = nn.LSTM(input_size=2,
                                    hidden_size=lstm_hidden_dim,
                                    num_layers=num_layers,
                                    bidirectional=bidir,
                                    batch_first=True,
                                    dropout=0.5)

        self.decoder = nn.LSTM(input_size=self.dir_number * lstm_hidden_dim,
                               hidden_size=embedding_dim,
                               num_layers=num_layers,
                               bidirectional=bidir,
                               batch_first=True,
                               dropout=0.5)

        # The linear layer that maps from hidden state space to pose space

        self.hidden2pose = nn.Linear(9 * self.dir_number * embedding_dim, target_size)

    def forward(self, scene: torch.Tensor) -> torch.Tensor:
        """
        :param scene: tensor of shape num_peds, history_size, data_dim
        :return: predicted poses for each agent at next timestep
        """
        bs = scene.shape[0]
        inp = scene

        lstm_out, hid = self.node_hist_encoder(inp)  # lstm_out shape num_peds, timestamps ,  2*hidden_dim
        current = scene[:, -1, :]  # num_people, data_dim
        np, data_dim = current.shape
        stacked = current.flatten().repeat(np).reshape(np, np * data_dim)
        deltas = (stacked - current.repeat(1, np)).reshape(np, np, data_dim)  # np, np, data_dim
        distruction, _ = self.edge_encoder(deltas)
        catted = torch.cat((lstm_out, distruction[:, -1:, :]), dim=1)
        decoded, _ = self.decoder(catted)
        dec_reshaped = decoded.reshape(bs, -1)
        tag_space = (scene.clone()[:, -1, :] + self.hidden2pose(dec_reshaped)).unsqueeze(1)
        return tag_space


class LstmEncDeltaAllHistStacked(nn.Module):
    """
    model that actually predicts delta movements (output last timestamp + predicted delta),
    with encoding person history and neighbors relative positions.
    """

    def __init__(self, lstm_hidden_dim, target_size, num_layers=1, embedding_dim=10, bidir=True):
        super(LstmEncDeltaAllHistStacked, self).__init__()
        self.name = "LstmEncDeltaAllHistStacked"
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_layers = num_layers
        self.dir_number = 1
        if bidir:
            self.dir_number = 2
        self.node_hist_encoder = nn.LSTM(input_size=2,
                                         hidden_size=lstm_hidden_dim,
                                         num_layers=num_layers,
                                         bidirectional=bidir,
                                         batch_first=True,
                                         dropout=0.5)

        self.edge_encoder = nn.LSTM(input_size=2,
                                    hidden_size=lstm_hidden_dim,
                                    num_layers=num_layers,
                                    bidirectional=bidir,
                                    batch_first=True,
                                    dropout=0.5)

        self.edge_seq_encoder = nn.LSTM(input_size=lstm_hidden_dim,
                                        hidden_size=lstm_hidden_dim,
                                        num_layers=num_layers,
                                        bidirectional=bidir,
                                        batch_first=True,
                                        dropout=0.5)

        self.decoder = nn.LSTM(input_size=2 * self.dir_number * lstm_hidden_dim,
                               hidden_size=embedding_dim,
                               num_layers=num_layers,
                               bidirectional=bidir,
                               batch_first=True,
                               dropout=0.5)

        # The linear layer that maps from hidden state space to pose space

        self.hidden2pose = nn.Linear(8 * self.dir_number * embedding_dim, target_size)

    def forward(self, scene: torch.Tensor) -> torch.Tensor:
        """
        :param scene: tensor of shape num_peds, history_size, data_dim
        :return: predicted poses for each agent at next timestep
        """
        bs = scene.shape[0]
        inp = scene

        lstm_out, hid = self.node_hist_encoder(inp)  # lstm_out shape num_peds, timestamps ,  2*hidden_dim
        current = scene.clone()  # num_people, data_dim
        np, seq, d_dim = scene.shape
        stacked = current.permute(1, 0, 2).reshape(seq, -1).repeat(1, np).reshape(seq, np, np * d_dim)
        deltas = (stacked - current.permute(1, 0, 2).repeat(1, 1, np))
        deltas = deltas.permute(1, 0, 2).reshape(np, seq, np, d_dim)
        distruction_hist = torch.zeros_like(lstm_out)
        for person in range(np):
            distruction, _ = self.edge_encoder(deltas[person])
            distruction_hist[person] = distruction[:, -1, :].clone()
        full_dist, _ = self.edge_seq_encoder(distruction_hist)
        catted = torch.cat((lstm_out, full_dist), dim=2)
        decoded, _ = self.decoder(catted)
        dec_reshaped = decoded.reshape(bs, -1)
        tag_space = (scene.clone()[:, -1, :] + self.hidden2pose(dec_reshaped)).unsqueeze(1)
        return tag_space


class LstmEncDeltaStackedVel(nn.Module):
    """
    model that actually predicts delta movements (output last timestamp + predicted delta),
    with encoding person history and neighbors relative positions.
    """

    def __init__(self, lstm_hidden_dim, target_size, num_layers=1, embedding_dim=10, bidir=True):
        super(LstmEncDeltaStackedVel, self).__init__()
        self.name = "LstmEncDeltaStackedVel"
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_layers = num_layers
        self.dir_number = 1
        if bidir:
            self.dir_number = 2
        self.node_hist_encoder = nn.LSTM(input_size=4,
                                         hidden_size=lstm_hidden_dim,
                                         num_layers=num_layers,
                                         bidirectional=bidir,
                                         batch_first=True,
                                         dropout=0.5)

        self.node_vel_hist_encoder = nn.LSTM(input_size=4,
                                             hidden_size=lstm_hidden_dim,
                                             num_layers=num_layers,
                                             bidirectional=bidir,
                                             batch_first=True,
                                             dropout=0.5)

        self.edge_encoder = nn.LSTM(input_size=4,
                                    hidden_size=lstm_hidden_dim,
                                    num_layers=num_layers,
                                    bidirectional=bidir,
                                    batch_first=True,
                                    dropout=0.5)

        self.decoder = nn.LSTM(input_size=self.dir_number * lstm_hidden_dim,
                               hidden_size=embedding_dim,
                               num_layers=num_layers,
                               bidirectional=bidir,
                               batch_first=True,
                               dropout=0.5)

        # The linear layer that maps from hidden state space to pose space

        self.hidden2pose = nn.Linear(9 * self.dir_number * embedding_dim, target_size)

    def forward(self, scene: torch.Tensor) -> torch.Tensor:
        """
        :param scene: tensor of shape num_peds, history_size, data_dim
        :return: predicted poses for each agent at next timestep
        """
        bs = scene.shape[0]
        inp = scene

        lstm_out, hid = self.node_hist_encoder(inp)  # lstm_out shape num_peds, timestamps ,  2*hidden_dim
        # lstm_out_vel, _ = self.node_vel_hist_encoder(inp[..., 2:4])
        current = scene[:, -1, :]  # num_people, data_dim
        np, data_dim = current.shape
        stacked = current.flatten().repeat(np).reshape(np, np * data_dim)
        deltas = (stacked - current.repeat(1, np)).reshape(np, np, data_dim)  # np, np, data_dim
        distruction, _ = self.edge_encoder(deltas)
        # catted = torch.cat((lstm_out+lstm_out_vel, distruction[:, -1:, :]), dim=1)
        catted = torch.cat((lstm_out, distruction[:, -1:, :]), dim=1)
        decoded, _ = self.decoder(catted)
        dec_reshaped = decoded.reshape(bs, -1)
        tag_space = (scene.clone()[:, -1, :2] + self.hidden2pose(dec_reshaped)).unsqueeze(1)
        return tag_space


class LstmEncDeltaStackedFullPred(nn.Module):
    """
    model that actually predicts delta movements (output last timestamp + predicted delta),
    with encoding person history and neighbors relative positions.
    """

    def __init__(self, lstm_hidden_dim, target_size, num_layers=1, embedding_dim=10, bidir=True, dropout_p=0.5):
        super(LstmEncDeltaStackedFullPred, self).__init__()
        self.name = "LstmEncDeltaStackedFullPred"
        self.embedding_dim = embedding_dim
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_layers = num_layers
        self.dir_number = 1
        if bidir:
            self.dir_number = 2
        self.node_hist_encoder = nn.LSTM(input_size=4,
                                         hidden_size=lstm_hidden_dim,
                                         num_layers=num_layers,
                                         bidirectional=bidir,
                                         batch_first=True,
                                         dropout=0.5)

        self.edge_encoder = nn.LSTM(input_size=2,
                                    hidden_size=lstm_hidden_dim,
                                    num_layers=num_layers,
                                    bidirectional=bidir,
                                    batch_first=True,
                                    dropout=0.5)

        self.gru = nn.GRUCell(2 * lstm_hidden_dim * self.dir_number + 2, 16)

        self.action = nn.Linear(2, 2)
        self.state = nn.Linear(2 * lstm_hidden_dim * self.dir_number, 16)

        # The linear layer that maps from hidden state space to pose space

        self.hidden2pose = nn.Linear(9 * self.dir_number * embedding_dim, target_size)

        self.proj_to_GMM_mus = nn.Linear(16, 2)
        self.proj_to_GMM_log_sigmas = nn.Linear(16, 2)
        self.proj_to_GMM_corrs = nn.Linear(16, 1)
        self.dropout_p = dropout_p

    def project_to_GMM_params(self, tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
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

        mus = F.dropout(self.proj_to_GMM_mus(tensor), self.dropout_p)
        log_sigmas = F.dropout(self.proj_to_GMM_log_sigmas(tensor), self.dropout_p)
        corrs = F.dropout(torch.tanh(self.proj_to_GMM_corrs(tensor)), self.dropout_p)
        return mus, log_sigmas, corrs

    def forward(self, scene: torch.Tensor):
        """
        :param scene: tensor of shape num_peds, history_size, data_dim
        :return: predicted poses for each agent at next timestep
        """
        bs = scene.shape[0]
        poses = scene[:, :, :2]
        pv = scene[:, :, :4]

        lstm_out, hid = self.node_hist_encoder(pv)  # lstm_out shape num_peds, timestamps ,  2*hidden_dim

        current_pose = scene[:, -1, :2]  # num_people, data_dim
        current_state = poses[:, -1, :]
        np, data_dim = current_pose.shape
        stacked = current_pose.flatten().repeat(np).reshape(np, np * data_dim)
        deltas = (stacked - current_pose.repeat(1, np)).reshape(np, np, data_dim)  # np, np, data_dim

        distruction, _ = self.edge_encoder(deltas)

        # catted = torch.cat((lstm_out+lstm_out_vel, distruction[:, -1:, :]), dim=1)
        catted = torch.cat((lstm_out[:, -1:, :], distruction[:, -1:, :]), dim=1)

        a_0 = F.dropout(self.action(current_state.reshape(bs, -1)), self.dropout_p)

        state = F.dropout(self.state(catted.reshape(bs, -1)), self.dropout_p)

        gauses = []
        inp = F.dropout(torch.cat((catted.reshape(bs, -1), a_0), dim=-1), self.dropout_p)
        for i in range(12):
            h_state = self.gru(inp.reshape(bs, -1), state)

            deltas, log_sigmas, corrs = self.project_to_GMM_params(h_state)
            deltas = torch.clamp(deltas, max=1.5, min=-1.5)
            mus = deltas + current_state
            current_state = mus.clone()
            variance = torch.clamp(torch.exp(log_sigmas).unsqueeze(1) ** 2, max=1e3)
            m_diag = variance * torch.eye(2).to(variance.device)
            sigma_xy = torch.clamp(torch.prod(torch.exp(log_sigmas), dim=-1), min=1e-8, max=1e3)
            t = (sigma_xy * corrs.squeeze()).reshape(-1, 1, 1)
            cov_matrix = m_diag  # + anti_diag
            gaus = MultivariateNormal(mus, cov_matrix)
            gauses.append(gaus)
            # a_t = gaus.rsample()
            # a_t = mus
            a_t = F.dropout(self.action(mus.reshape(bs, -1)), self.dropout_p)
            state = h_state
            inp = F.dropout(torch.cat((catted.reshape(bs, -1), a_t), dim=-1), self.dropout_p)

        return gauses




class LstmEncDeltaStackedFullPredMultyGaus(nn.Module):
    """
    model that actually predicts delta movements (but output last timestamp + predicted deltas),
    with encoding person history and neighbors relative positions.
    """

    def __init__(self, lstm_hidden_dim, num_layers=1, bidir=True, dropout_p=0.5, num_modes=20):
        super(LstmEncDeltaStackedFullPredMultyGaus, self).__init__()
        self.name = "LstmEncDeltaStackedFullPredMultyGaus"
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
                                         batch_first=True,
                                         dropout=0.5)

        self.node_hist_encoder_vel = nn.LSTM(input_size=2,
                                         hidden_size=lstm_hidden_dim,
                                         num_layers=num_layers,
                                         bidirectional=bidir,
                                         batch_first=True,
                                         dropout=0.5)

        self.node_hist_encoder_acc = nn.LSTM(input_size=2,
                                             hidden_size=lstm_hidden_dim,
                                             num_layers=num_layers,
                                             bidirectional=bidir,
                                             batch_first=True,
                                             dropout=0.5)

        self.node_hist_encoder_poses = nn.LSTM(input_size=2,
                                             hidden_size=lstm_hidden_dim,
                                             num_layers=num_layers,
                                             bidirectional=bidir,
                                             batch_first=True,
                                             dropout=0.5)


        self.edge_encoder = nn.LSTM(input_size=2,
                                    hidden_size=lstm_hidden_dim,
                                    num_layers=num_layers,
                                    bidirectional=bidir,
                                    batch_first=True,
                                    dropout=0.5)

        self.gru = nn.GRUCell(2 * lstm_hidden_dim * self.dir_number + 2, num_modes*2)

        self.action = nn.Linear(2, 2)
        self.state = nn.Linear(2 * lstm_hidden_dim * self.dir_number, num_modes*2)

        self.proj_to_GMM_log_pis = nn.Linear(num_modes*2, num_modes)
        self.proj_to_GMM_mus = nn.Linear(num_modes*2, num_modes*2)
        self.proj_to_GMM_log_sigmas = nn.Linear(num_modes*2, num_modes*2)
        self.proj_to_GMM_corrs = nn.Linear(num_modes*2, num_modes)
        self.dropout_p = dropout_p

    def project_to_GMM_params(self, tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
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
        log_pis =  F.dropout(self.proj_to_GMM_log_pis(tensor), self.dropout_p)
        mus = F.dropout(self.proj_to_GMM_mus(tensor), self.dropout_p)
        log_sigmas = F.dropout(self.proj_to_GMM_log_sigmas(tensor), self.dropout_p)
        corrs = F.dropout(torch.tanh(self.proj_to_GMM_corrs(tensor)), self.dropout_p)
        return log_pis, mus, log_sigmas, corrs

    def forward(self, scene: torch.Tensor):
        """
        :param scene: tensor of shape num_peds, history_size, data_dim
        :return: predicted poses distributions for each agent at next 12 timesteps
        """
        bs = scene.shape[0]
        poses = scene[:, :, :2]
        pv = scene[:, :, 2:6]
        vel = scene[:, :, 2:4]
        acc = scene[:, :, 4:6]
        pav = scene[:, :, :6]

        # lstm_out, hid = self.node_hist_encoder(pav)  # lstm_out shape num_peds, timestamps ,  2*hidden_dim
        lstm_out_acc, hid = self.node_hist_encoder_acc(acc)  # lstm_out shape num_peds, timestamps ,  2*hidden_dim
        lstm_out_vell, hid = self.node_hist_encoder_vel(vel)  # lstm_out shape num_peds, timestamps ,  2*hidden_dim
        lstm_out_poses, hid = self.node_hist_encoder_poses(poses)
        lstm_out = lstm_out_vell + lstm_out_poses + lstm_out_acc
        # lstm_out = lstm_out_poses  # + lstm_out_poses

        current_pose = scene[:, -1, :2]  # num_people, data_dim
        current_state = poses[:, -1, :]
        np, data_dim = current_pose.shape
        stacked = current_pose.flatten().repeat(np).reshape(np, np * data_dim)
        deltas = (stacked - current_pose.repeat(1, np)).reshape(np, np, data_dim)  # np, np, data_dim

        distruction, _ = self.edge_encoder(deltas)
        catted = torch.cat((lstm_out[:, -1:, :], distruction[:, -1:, :]), dim=1)
        a_0 = F.dropout(self.action(current_state.reshape(bs, -1)), self.dropout_p)
        state = F.dropout(self.state(catted.reshape(bs, -1)), self.dropout_p)

        current_state = current_state.unsqueeze(1)
        gauses = []
        inp = F.dropout(torch.cat((catted.reshape(bs, -1), a_0), dim=-1), self.dropout_p)
        for i in range(12):
            h_state = self.gru(inp.reshape(bs, -1), state)

            log_pis, deltas, log_sigmas, corrs = self.project_to_GMM_params(h_state)
            deltas = torch.clamp(deltas, max=1.5, min=-1.5)

            log_pis = log_pis.reshape(bs, -1)
            log_pis = log_pis - torch.logsumexp(log_pis, dim=-1, keepdim=True)
            deltas = deltas.reshape(bs, -1, 2)
            log_sigmas = log_sigmas.reshape(bs, -1, 2)
            corrs = corrs.reshape(bs, -1, 1)

            mus = deltas + current_state
            current_state = mus
            variance = torch.clamp(torch.exp(log_sigmas).unsqueeze(2) ** 2, max=1e3)

            m_diag = variance * torch.eye(2).to(variance.device)
            sigma_xy = torch.clamp(torch.prod(torch.exp(log_sigmas), dim=-1), min=1e-8, max=1e3)

            mix = D.Categorical(log_pis)
            comp = D.MultivariateNormal(mus, m_diag)
            gmm = D.MixtureSameFamily(mix, comp)
            t = (sigma_xy * corrs.squeeze()).reshape(-1, 1, 1)
            cov_matrix = m_diag  # + anti_diag
            gauses.append(gmm)
            a_t = gmm.sample()  # possible grad problems?
            a_tt = F.dropout(self.action(a_t.reshape(bs, -1)), self.dropout_p)
            state = h_state
            inp = F.dropout(torch.cat((catted.reshape(bs, -1), a_tt), dim=-1), self.dropout_p)

        return gauses


class AttentionDecoder(nn.Module):

    def __init__(self, hidden_size, output_size, input_size=30, obs_len=2, device= None):
        super(AttentionDecoder, self).__init__()
        if device is None:
            device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        self.device = device
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.attn = nn.Linear(output_size + input_size, 1)
        self.lstm = nn.LSTM(hidden_size + input_size,
                            output_size, batch_first=True)  # if we are using embedding hidden_size should be added with embedding of vocab size
        self.final = nn.Linear(output_size, output_size)

    def init_hidden(self, bs=1, num_l=1):
        return (torch.zeros(num_l, bs, self.output_size).to(self.device),
                torch.zeros(num_l, bs, self.output_size).to(self.device))

    def forward(self, decoder_hidden, encoder_outputs, input):
        weights = []
        # print(decoder_hidden[0].shape)
        # print("encoder_outputs ", encoder_outputs.shape)
        # print(input.shape)
        weights = self.attn(torch.cat((decoder_hidden[0].permute(1, 0, 2).repeat(1, input.shape[1], 1), input), dim=2))
        # print(weights.shape)

        normalized_weights = F.softmax(weights)

        # print("normalized_weights ", normalized_weights.shape)
        # print(normalized_weights.shape)
        # print(encoder_outputs.shape)

        attn_applied = torch.bmm(normalized_weights,
                                 encoder_outputs)
        # print("attn_applied ", attn_applied.shape)
        # print("input[0].shape", input[0].shape)
        input_lstm = torch.cat((attn_applied, input),
                               dim=2)  # if we are using embedding, use embedding of input here instead
        # print(input_lstm.shape)
        output, hidden = self.lstm(input_lstm, decoder_hidden)

        output = self.final(output)

        return output, hidden, normalized_weights


class LstmEncDWithAtt(nn.Module):
    """

    """

    def __init__(self, lstm_hidden_dim, num_layers=1, bidir=True, dropout_p=0.5, num_modes=20):
        super(LstmEncDWithAtt, self).__init__()
        self.name = "LstmEncDWithAtt"
        self.num_modes = num_modes
        self.embedding_dim = 0
        self.lstm_hidden_dim = lstm_hidden_dim
        self.num_layers = num_layers
        self.dir_number = 1
        if bidir:
            self.dir_number = 2
        self.node_hist_encoder = nn.LSTM(input_size=2,
                                         hidden_size=lstm_hidden_dim,
                                         num_layers=num_layers,
                                         bidirectional=bidir,
                                         batch_first=True,
                                         dropout=0.5)

        self.node_hist_encoder_vel = nn.LSTM(input_size=2,
                                         hidden_size=lstm_hidden_dim,
                                         num_layers=num_layers,
                                         bidirectional=bidir,
                                         batch_first=True,
                                         dropout=0.5)

        self.node_hist_encoder_acc = nn.LSTM(input_size=2,
                                             hidden_size=lstm_hidden_dim,
                                             num_layers=num_layers,
                                             bidirectional=bidir,
                                             batch_first=True,
                                             dropout=0.5)

        self.node_hist_encoder_poses = nn.LSTM(input_size=2,
                                             hidden_size=lstm_hidden_dim,
                                             num_layers=num_layers,
                                             bidirectional=bidir,
                                             batch_first=True,
                                             dropout=0.5)


        self.edge_encoder = nn.LSTM(input_size=2,
                                    hidden_size=lstm_hidden_dim,
                                    num_layers=num_layers,
                                    bidirectional=bidir,
                                    batch_first=True,
                                    dropout=0.5)

        self.gru = nn.GRUCell(2 * lstm_hidden_dim * self.dir_number + 2, num_modes*2)

        self.action = nn.Linear(2, 2)
        self.state = nn.Linear(2 * lstm_hidden_dim * self.dir_number, num_modes*2)

        self.proj_to_GMM_log_pis = nn.Linear(num_modes*2, num_modes)
        self.proj_to_GMM_mus = nn.Linear(num_modes*2, num_modes*2)
        self.proj_to_GMM_log_sigmas = nn.Linear(num_modes*2, num_modes*2)
        self.proj_to_GMM_corrs = nn.Linear(num_modes*2, num_modes)
        self.dropout_p = dropout_p
        self.att = AttentionDecoder(hidden_size=lstm_hidden_dim * self.dir_number, output_size=self.dir_number*lstm_hidden_dim, input_size=50, obs_len=8)

    def project_to_GMM_params(self, tensor) -> (torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor):
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
        log_pis = F.dropout(self.proj_to_GMM_log_pis(tensor), self.dropout_p)
        mus = F.dropout(self.proj_to_GMM_mus(tensor), self.dropout_p)
        log_sigmas = F.dropout(self.proj_to_GMM_log_sigmas(tensor), self.dropout_p)
        corrs = F.dropout(torch.tanh(self.proj_to_GMM_corrs(tensor)), self.dropout_p)
        return log_pis, mus, log_sigmas, corrs

    def forward(self, scene: torch.Tensor):
        """
        :param scene: tensor of shape num_peds, history_size, data_dim
        :return: predicted poses distributions for each agent at next 12 timesteps
        """
        bs = scene.shape[0]
        poses = scene[:, :, :2]
        pv = scene[:, :, 2:6]
        vel = scene[:, :, 2:4]
        acc = scene[:, :, 4:6]
        pav = scene[:, :, :6]

        lstm__poses_out, _ = self.node_hist_encoder_poses(poses)  # lstm_out shape num_peds, timestamps ,  2*hidden_dim
        lstm_out_acc, hid = self.node_hist_encoder_acc(acc)  # lstm_out shape num_peds, timestamps ,  2*hidden_dim
        lstm_out_vell, hid = self.node_hist_encoder_vel(vel)  # lstm_out shape num_peds, timestamps ,  2*hidden_dim
        # lstm_out_poses, hid = self.node_hist_encoder_poses(poses)
        lstm_out = lstm_out_vell + lstm_out_acc  # + lstm_out_poses

        current_state = poses[:, -1, :]
        # np, data_dim = current_pose.shape
        bs, seq, data_dim = poses.shape
        stacked = poses.permute(1, 0, 2).reshape(seq, -1).repeat(1, bs).reshape(seq, bs, bs * data_dim)
        deltas = (stacked - poses.permute(1, 0, 2).repeat(1, 1, bs))
        deltas = deltas.permute(1, 0, 2).reshape(bs, seq, bs, data_dim)
        deltas_flat = deltas.reshape(deltas.shape[0], deltas.shape[1], -1).cuda()
        max_size = 50 # TODO: fix
        prep_for_deltas = torch.zeros(bs, seq, 50).cuda()
        if deltas_flat.shape[2] >= max_size:
            prep_for_deltas = deltas_flat[:, :, :max_size]
        else:
            prep_for_deltas[:, :, :deltas_flat.shape[2]] = deltas_flat
        at_hidden = self.att.init_hidden(bs=bs)
        for i in range(8):
            at_output, at_hidden, at_normalized_weights = self.att(at_hidden, lstm__poses_out[:, i:i+1, :],
                                                                   prep_for_deltas[:, i:i+1, :])
        # current_pose = scene[:, -1, :2]  # num_people, data_dim
        # stacked = current_pose.flatten().repeat(np).reshape(np, np * data_dim)
        # deltas = (stacked - current_pose.repeat(1, np)).reshape(np, np, data_dim)  # np, np, data_dim

        # distruction, _ = self.edge_encoder(deltas, poses, poses)
        catted = torch.cat((lstm_out[:, -1:, :], at_output[:, -1:, :]), dim=2)
        a_0 = F.dropout(self.action(current_state.reshape(bs, -1)), self.dropout_p)
        state = F.dropout(self.state(catted.reshape(bs, -1)), self.dropout_p)

        current_state = current_state.unsqueeze(1)
        gauses = []
        inp = F.dropout(torch.cat((catted.reshape(bs, -1), a_0), dim=-1), self.dropout_p)
        for i in range(12):
            h_state = self.gru(inp.reshape(bs, -1), state)

            log_pis, deltas, log_sigmas, corrs = self.project_to_GMM_params(h_state)
            deltas = torch.clamp(deltas, max=1.5, min=-1.5)

            log_pis = log_pis.reshape(bs, -1)
            log_pis = log_pis - torch.logsumexp(log_pis, dim=-1, keepdim=True)
            deltas = deltas.reshape(bs, -1, 2)
            log_sigmas = log_sigmas.reshape(bs, -1, 2)
            corrs = corrs.reshape(bs, -1, 1)

            mus = deltas + current_state
            current_state = mus
            variance = torch.clamp(torch.exp(log_sigmas).unsqueeze(2) ** 2, max=1e3)

            m_diag = variance * torch.eye(2).to(variance.device)
            sigma_xy = torch.clamp(torch.prod(torch.exp(log_sigmas), dim=-1), min=1e-8, max=1e3)

            mix = D.Categorical(log_pis)
            comp = D.MultivariateNormal(mus, m_diag)
            gmm = D.MixtureSameFamily(mix, comp)
            t = (sigma_xy * corrs.squeeze()).reshape(-1, 1, 1)
            cov_matrix = m_diag  # + anti_diag
            gauses.append(gmm)
            a_t = gmm.sample()  # possible grad problems?
            state = h_state
            inp = F.dropout(torch.cat((catted.reshape(bs, -1), a_t), dim=-1), self.dropout_p)
        return gauses

class OneLayer(nn.Module):
    def __init__(self):
        super(OneLayer, self).__init__()
        self.simple_linear = nn.Linear(16, 2)

    #         self.simple_linear1 = nn.Linear(20, 40)
    #         self.simple_linear2 = nn.Linear(40, 8)

    def forward(self, self_pose, others_pose):
        self_pose = self_pose.reshape(-1, 16)

        out = self.simple_linear(self_pose)

        return out.reshape(1, 1, 2)



