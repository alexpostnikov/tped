import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.distributions as D
from torch.autograd import Variable


class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, n_head, d_model, d_k, d_v, dropout=0.1):
        super().__init__()

        self.n_head = n_head
        self.d_k = d_k
        self.d_v = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_ks = nn.Linear(d_model, n_head * d_k, bias=False)
        self.w_vs = nn.Linear(d_model, n_head * d_v, bias=False)
        self.fc = nn.Linear(n_head * d_v, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.d_k, self.d_v, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q, attn = self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attn


class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output, attn




def to_one_hot(y, n_dims=None):
    """ Take integer y (tensor or variable) with n dims and convert it to 1-hot representation with n+1 dims. """
    y_tensor = y.data if isinstance(y, Variable) else y
    y_tensor = y_tensor.type(torch.LongTensor).view(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    y_one_hot = y_one_hot.view(*y.shape, -1)
    return Variable(y_one_hot) if isinstance(y, Variable) else y_one_hot




class CvaeAtt(nn.Module):
    """
    model that actually predicts delta movements (but output last timestamp + predicted deltas),
    with encoding person history and neighbors relative positions.
    """

    def __init__(self, lstm_hidden_dim, num_layers=1, bidir=True, dropout_p=0.5, num_modes=20):
        super(CvaeAtt, self).__init__()
        self.name = "CvaeAtt"
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

        self.node_future_encoder = nn.LSTM(input_size=6,
                                    hidden_size=lstm_hidden_dim,
                                    num_layers=num_layers,
                                    bidirectional=bidir,
                                    batch_first=True,
                                    dropout=0.5)

        self.att = MultiHeadAttention(8, 2, 10, 10)

        self.gru = nn.GRUCell(2 * lstm_hidden_dim * self.dir_number + 2, num_modes*2)

        self.action = nn.Linear(2, 2)
        self.state = nn.Linear(2 * lstm_hidden_dim * self.dir_number, num_modes*2)
        self.proj_p_to_log_pis = nn.Linear(lstm_hidden_dim*4, num_modes)
        self.proj_to_GMM_log_pis = nn.Linear(lstm_hidden_dim*6, num_modes)
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
        # log_pis =  F.dropout(self.proj_to_GMM_log_pis(tensor), self.dropout_p)
        mus = F.dropout(self.proj_to_GMM_mus(tensor), self.dropout_p)
        log_sigmas = F.dropout(self.proj_to_GMM_log_sigmas(tensor), self.dropout_p)
        corrs = F.dropout(torch.tanh(self.proj_to_GMM_corrs(tensor)), self.dropout_p)
        return None, mus, log_sigmas, corrs


    def sample_q(self):
        pass

    def sample_p(self):
        pass


    def obtain_encoded_tensors(self, scene: torch.Tensor):
        bs = scene.shape[0]
        poses = scene[:, :8, :2]
        pv = scene[:, :8, 2:6]
        vel = scene[:, :8, 2:4]
        acc = scene[:, :8, 4:6]
        pav = scene[:, :8, :6]
        future = scene[:, 8:, :6]

        # lstm_out, hid = self.node_hist_encoder(pav)  # lstm_out shape num_peds, timestamps , 2*hidden_dim
        lstm_out_acc, hid = self.node_hist_encoder_acc(acc)  # lstm_out shape num_peds, timestamps,  2*hidden_dim
        lstm_out_vell, hid = self.node_hist_encoder_vel(vel)  # lstm_out shape num_peds, timestamps,  2*hidden_dim
        lstm_out_poses, hid = self.node_hist_encoder_poses(poses)
        lstm_out = lstm_out_vell + lstm_out_poses + lstm_out_acc
        future_enc, hid = self.node_future_encoder(future)
        y_e = future_enc[:, -1, :]
        # lstm_out = lstm_out_poses  # + lstm_out_poses

        current_pose = scene[:, 7, :2]  # num_people, data_dim

        np, data_dim = current_pose.shape
        stacked = current_pose.flatten().repeat(np).reshape(np, np * data_dim)
        deltas = (stacked - current_pose.repeat(1, np)).reshape(np, np, data_dim)  # np, np, data_dim
        att_deltas1 = self.att(deltas, deltas, deltas)[0]
        att_deltas2 = self.att(att_deltas1, att_deltas1, att_deltas1)[0]
        att_deltas = self.att(att_deltas2, att_deltas2, att_deltas2)[0]

        distruction, _ = self.edge_encoder(att_deltas)
        encoded_history = torch.cat((lstm_out[:, -1:, :], distruction[:, -1:, :]), dim=1)

        return encoded_history, y_e

    def sample(self, p_distrib: D.Categorical):
        pass
        z = p_distrib.sample()
        return z

    def encoder(self, x, y_e=None, train=True):
        pass
        # q_distrib -> onehot?
        # p_distrib -> onehot?
        # kl dist
        # sample z
        bs = x.shape[0]
        p_distrib = D.Categorical(logits=F.dropout(self.proj_p_to_log_pis(x.reshape(bs,-1)), self.dropout_p))
        kl = 0
        q_distrib = None

        if train:
            h = torch.cat((x.reshape(bs, -1), y_e), dim=1)
            q_distrib = D.Categorical(logits=F.dropout(self.proj_to_GMM_log_pis(h), self.dropout_p))
            z = self.sample(q_distrib)
            kl_separated = D.kl_divergence(q_distrib, p_distrib)
            kl_lower_bounded = torch.clamp(kl_separated, min=0.1, max=1e5)
            kl = torch.sum(kl_lower_bounded)
        else:
            z = self.sample(p_distrib)
        return p_distrib, q_distrib, z, kl

    def decoder(self, z, encoded_history, current_state, y_e=None, train=False):
        pass

        bs = encoded_history.shape[0]
        a_0 = F.dropout(self.action(current_state.reshape(bs, -1)), self.dropout_p)
        state = F.dropout(self.state(encoded_history.reshape(bs, -1)), self.dropout_p)

        current_state = current_state.unsqueeze(1)
        gauses = []
        inp = F.dropout(torch.cat((encoded_history.reshape(bs, -1), a_0), dim=-1), self.dropout_p)
        for i in range(12):
            h_state = self.gru(inp.reshape(bs, -1), state)

            _, deltas, log_sigmas, corrs = self.project_to_GMM_params(h_state)
            deltas = torch.clamp(deltas, max=1.5, min=-1.5)
            deltas = deltas.reshape(bs, -1, 2)
            log_sigmas = log_sigmas.reshape(bs, -1, 2)
            corrs = corrs.reshape(bs, -1, 1)

            mus = deltas + current_state
            current_state = mus
            variance = torch.clamp(torch.exp(log_sigmas).unsqueeze(2) ** 2, max=1e3)

            m_diag = variance * torch.eye(2).to(variance.device)
            sigma_xy = torch.clamp(torch.prod(torch.exp(log_sigmas), dim=-1), min=1e-8, max=1e3)

            if train:
                log_pis = z.reshape(bs, 1) * torch.ones(bs, self.num_modes).cuda()

            else:
                log_pis = to_one_hot(z, n_dims=self.num_modes).cuda()
            log_pis = log_pis - torch.logsumexp(log_pis, dim=-1, keepdim=True)
            mix = D.Categorical(logits=log_pis)
            comp = D.MultivariateNormal(mus, m_diag)
            gmm = D.MixtureSameFamily(mix, comp)
            t = (sigma_xy * corrs.squeeze()).reshape(-1, 1, 1)
            cov_matrix = m_diag  # + anti_diag
            gauses.append(gmm)
            a_t = gmm.sample()  # possible grad problems?
            a_tt = F.dropout(self.action(a_t.reshape(bs, -1)), self.dropout_p)
            state = h_state
            inp = F.dropout(torch.cat((encoded_history.reshape(bs, -1), a_tt), dim=-1), self.dropout_p)

        return gauses


    def loss(self, gmm, kl):
        # ELBO = log_likelihood - self.kl_weight * kl + 1. * mutual_inf_p
        pass

    def forward(self, scene: torch.Tensor, train=False):
        """
        :param scene: tensor of shape num_peds, history_size, data_dim
        :return: predicted poses distributions for each agent at next 12 timesteps
        """

        encoded_history, enc_future = self.obtain_encoded_tensors(scene)
        p_distrib, q_distrib, z, kl = self.encoder(encoded_history, enc_future, train)
        gmm = self.decoder(z, encoded_history, scene[:, 7, :2], train=train)
        if train:
            return gmm, kl
        else:
            return gmm




if __name__ == "__main__":
    attention = MultiHeadAttention(10, 5, 5, 5)
    i = torch.rand(10, 5, 5)
    attention(i, i, i)