import math
import time
from datetime import datetime
from typing import Tuple, Union
import os

import torch
import torch.optim as optim
from dataloader import Dataset_from_pkl, is_filled
from model import LstmEncDeltaStacked, LstmEncDeltaStackedVel, LstmEncDeltaStackedFullPred, LSTM_single, \
    LSTM_enc_delta_wo_emb, LstmEncDeltaStacked, LstmEncDeltaAllHistStacked

from model_cvae import Cvae, CvaeFuture
from model import LstmEncDeltaStackedFullPredMultyGaus, LstmEncDWithAtt
from torch.utils.tensorboard import SummaryWriter
import torch.distributions as D
from train_with_future import get_ade_fde_vel
import shutil
import matplotlib.pyplot as plt
from visualize import plot_traj

from utils import compare_prediction_gt


class Ansamble:
    def __init__(self, models):
        self.models = models

    def __call__(self, scene: torch.Tensor, train=False):
        gmms = []
        for model in self.models:
            gmm = model(scene)
            gmms.append(gmm)

        combined_gmm = []
        for timestamp in range(12):
            logits = torch.cat([gmms[i][timestamp].mixture_distribution.logits for i in range(len(gmms))], dim=1)
            mus = torch.cat([gmms[i][timestamp].component_distribution.mean for i in range(len(gmms))], dim=1)
            variance = torch.cat([gmms[i][timestamp].component_distribution.variance for i in range(len(gmms))], dim=1)
            m_diag = variance.unsqueeze(2) * torch.eye(2).to(variance.device)
            mix = D.Categorical(logits)
            comp = D.MultivariateNormal(mus, m_diag)
            combined_gmm.append(D.MixtureSameFamily(mix, comp))
        return combined_gmm

    def eval(self):
        for model in self.models:
            model.eval()

    def train(self):
        for model in self.models:
            model.train()







if __name__ == "__main__":

    ansamble_path = "./models/ansamble2/"

    pathes = [os.path.join(ansamble_path, o)+"/model.pth" for o in os.listdir(ansamble_path)
                if os.path.isdir(os.path.join(ansamble_path, o))]

    training_set = Dataset_from_pkl("/home/robot/repos/trajectory-prediction/processed/", data_files=["eth_train.pkl"])
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=1, shuffle=True)

    test_set = Dataset_from_pkl("/home/robot/repos/trajectory-prediction/processed/", data_files=["eth_test.pkl"])
    test_generator = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # print(device)
    #
    # models = []
    #
    # for path in pathes:
    #     model = CvaeFuture(lstm_hidden_dim=64, num_layers=1, bidir=True, dropout_p=0.0, num_modes=30).to(device)
    #     model.load_state_dict(torch.load(path))
    #     models.append(model)
    #
    # input = torch.rand(1, 20, 6).to(device)
    # ansamble = Ansamble(models)
    #
    # combined_gmm = ansamble(input)

    # start = time.time()
    # with torch.no_grad():
    #     for counter, model in enumerate(models):
    #         ade, fde, nll = get_ade_fde_vel(test_generator, model)
    #         print("models", counter, ": ", ade, fde, nll)
    #     ade, fde, nll = get_ade_fde_vel(test_generator, ansamble)
    #     print("ansamble: ", ade, fde, nll)

    model = CvaeFuture(lstm_hidden_dim=64, num_layers=1, bidir=True, dropout_p=0.0, num_modes=30).to(device)
    model.load_state_dict(torch.load(
        "/home/robot/repos/trajectory-prediction/tb/CvaeAtt0.0005_hd_64_ed_0_nl_1@14.08.2020-18:13:15/CvaeAtt0.0005_hd_64_ed_0_nl_1@14.08.2020-18:13:15+best.pth"))
    with torch.no_grad():

        ade, fde, nll = get_ade_fde_vel(test_generator, model)
        print("model: ", ade, fde, nll)
        # ade, fde, nll = get_ade_fde_vel(test_generator, ansamble)
        # print("ansamble: ", ade, fde, nll)

