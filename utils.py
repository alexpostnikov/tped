from typing import Tuple

import torch
import numpy as np
from scipy.stats import gaussian_kde

from dataloader_parallel import is_filled
from datetime import datetime
import os
from torch.utils.tensorboard import SummaryWriter

magic_number = 0

def load_model(model_path):
    print('Loading model from \n' + model_path)
    model = torch.load(model_path)
    print('Loaded!\n')
    return model


def calc_ade(predicted_trajs, gt_traj):
    error = np.linalg.norm(predicted_trajs - gt_traj, axis=-1)
    ade = np.mean(error, axis=-1)
    return ade


def calc_fde(predicted, gt):
    final_error = np.linalg.norm(predicted - gt)
    return final_error


def calc_kde_nll(predicted_trajs, gt_traj):
    kde_ll = 0.
    log_pdf_lower_bound = -20
    num_timesteps = gt_traj.shape[0]

    for timestep in range(num_timesteps):
        try:
            kde = gaussian_kde(predicted_trajs[timestep].T)
            pdf = np.clip(kde.logpdf(gt_traj[timestep].T), a_min=log_pdf_lower_bound, a_max=None)[0]
            kde_ll += pdf / (num_timesteps)
        except np.linalg.LinAlgError:
            kde_ll = np.nan

    return -kde_ll


def is_data_enough(data):
    for step in data:
        for coord in step:
            if coord == magic_number:
                return False
    return True


def compare_prediction_gt(prediction, gt):
    # gt         -> 1, num_ped, 20, 2
    # prediction -> 1, num_ped, 12, 2
    error_dict = {'id': list(), 'ade': list(), 'fde': list(), 'kde': list()}
    for num_ped in range(len(prediction)):
        # check data

        if not is_data_enough(gt[num_ped]):
            continue
        # id
        error_dict['id'].append(num_ped)
        # ade
        ade = calc_ade(prediction[num_ped], gt[num_ped, 8:])
        error_dict['ade'].append(ade)
        # fde
        fde = calc_fde(prediction[num_ped, -1], gt[num_ped, -1])
        error_dict['fde'].append(fde)
        # kde_nll
        kde = calc_kde_nll(prediction[num_ped], gt[num_ped, 8:])
        error_dict['kde'].append(kde)
    # error_dict -> {'id':[num_ped],'ade': [num_ped], 'fde': [num_ped], 'kde': [num_ped]}
    return error_dict


def get_batch_is_filled_mask(batch: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """
    :param batch:  batch with shape bs, num_peds, seq, data_shape, currently works only with bs = 1
    :return: mask shape num_people, seq, data_shape with 0 filled data if person is not fully observable during seq
    """
    # assert batch.shape[0] == 1

    num_peds = batch.shape[0]
    mask = torch.zeros(num_peds, 12, 2)
    full_peds = 0
    for ped in range(num_peds):
        if is_filled(batch[ped]):
            mask[ped] = torch.ones(12, 2)
            full_peds += 1
    return mask, full_peds



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