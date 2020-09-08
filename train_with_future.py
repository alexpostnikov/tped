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
from cvae_att import CvaeAtt
from model import LstmEncDeltaStackedFullPredMultyGaus, LstmEncDWithAtt
import torch.distributions as D

from torch.utils.tensorboard import SummaryWriter
import shutil
import matplotlib.pyplot as plt
from visualize import plot_traj

from typing import List
from utils import compare_prediction_gt



def plot(predictions, gt):
    num_peds = gt.shape[1]
    for ped_num in range(num_peds):
        if is_filled(gt[0, ped_num, :8, :]):
            fig = plt.figure()
            ax1 = fig.add_subplot(2, 1, 1)
            ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
            plot_traj(gt[0, :, :8, 2:4].detach().cpu(), ax=ax1, color="blue")
            plot_traj(predictions[ped_num:ped_num + 1, :, :].detach().cpu(), ax2)
            plot_traj(gt[0, ped_num:ped_num + 1, 8:, 2:4].detach().cpu(), ax2, color="black")
            plot_traj(gt[0, ped_num:ped_num + 1, :8, 2:4].detach().cpu(), ax2, color="blue")
            plt.show()


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
    means = torch.cat(means).reshape(-1,2)
    return means


def get_ml_prediction_multiple_timestamps(gmm: List[D.MixtureSameFamily]):
    means = []
    for single_gmm in gmm:
        means.append(get_ml_prediction_single_timestamp(single_gmm))
    return torch.stack(means).permute(1,0,2)

def get_mean_prediction_multiple_timestamps(gmm: List[D.MixtureSameFamily]):
    mean_predictions =torch.cat([gmm[i].mean for i in range(12)]).reshape(12, -1, 2).permute(1, 0, 2)
    return mean_predictions

def get_ade_fde_vel(generator: torch.utils.data.DataLoader, model: torch.nn.Module, limit: int = 1e10) -> Tuple[
    int, int, int]:
    """
    :param generator: torch generator to get data
    :param model:   torch module predicting poses. input shape are [numped, history, 2]
    :param limit: limit number of processed batches. Default - unlimited
    :return: tuple of ade, fde for given generator and limit of batches
    """
    ade = []
    fde = []
    nll = 0
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    for batch_id, local_batch in enumerate(generator):

        if batch_id > limit:
            break
        if local_batch.shape[1] < 1:
            continue


        local_batch = local_batch.to(device)
        gt = local_batch.clone()
        mask, full_peds = get_batch_is_filled_mask(gt)
        mask = mask.to(device)
        # local_batch[0, :, 8:, :] = torch.zeros_like(local_batch[0, :, 8:, :])
        # num_peds = local_batch.shape[1]

        prediction = model(local_batch[0, :, :, 2:8])

        gt_prob = torch.cat(([prediction[i].log_prob(gt[0, :, 8 + i, 2:4]) for i in range(12)])).reshape(-1, 12)
        nll += -torch.sum(gt_prob * mask[:, :, 0])
        predictions = get_mean_prediction_multiple_timestamps(prediction)
        if 0:
            predictions = []
            for i in range(10):
                # model.eval()
                prediction = model(local_batch[0, :, :, 2:8])
                p = get_mean_prediction_multiple_timestamps(prediction)
                predictions.append(p)
            model.train()
            predictions = torch.mean(torch.stack(predictions), dim=0)
        # predictions = get_ml_prediction_multiple_timestamps(prediction)
        if torch.any(predictions != predictions):
            print("Warn! nan pred")
            continue

        metrics = compare_prediction_gt(predictions.detach().cpu(), gt.detach().cpu()[0][:, :, 2:4])
        for local_ade in metrics["ade"]:
            ade.append(local_ade)
        for local_fde in metrics["fde"]:
            fde.append(local_fde)

    nll = nll.item() / len(fde)
    ade = sum(ade) / len(ade)
    fde = sum(fde) / len(fde)

    return ade, fde, nll


def get_batch_is_filled_mask(batch: torch.Tensor) -> Tuple[torch.Tensor, int]:
    """
    :param batch:  batch with shape bs, num_peds, seq, data_shape, currently works only with bs = 1
    :return: mask shape num_people, seq, data_shape with 0 filled data if person is not fully observable during seq
    """
    assert batch.shape[0] == 1

    num_peds = batch.shape[1]
    mask = torch.zeros(num_peds, 12, 2)
    full_peds = 0
    for ped in range(num_peds):
        if is_filled(batch[0][ped]):
            mask[ped] = torch.ones(12, 2)
            full_peds += 1
    return mask, full_peds


def train_pose_vel(model: torch.nn.Module, training_generator: torch.utils.data.DataLoader,
                   test_generator: torch.utils.data.DataLoader, num_epochs: int, device: torch.device,
                   lr: Union[int, float] = 0.02, limit: Union[int, float] = 10e7, validate: bool = True):
    """

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

    optimizer = optim.Adam(model.parameters(), lr=lr)
    drop_every_epochs = 10
    drop_rate = 1
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, drop_every_epochs,
                                                drop_rate)  # drop_every_epochs epochs drop by drop_rate lr

    writer, experiment_name, save_model_path, folder_path = setup_experiment(model.name + str(lr) + "_hd_"
                                                                             + str(model.lstm_hidden_dim)
                                                                             + "_ed_" + str(model.embedding_dim)
                                                                             + "_nl_" + str(model.num_layers))

    dir_path = os.path.dirname(os.path.realpath(__file__))
    shutil.copyfile(f"{dir_path}/train.py", f"{folder_path}/train_with_future.py")
    shutil.copyfile(f"{dir_path}/model.py", f"{folder_path}/model.py")
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
        for batch_id, local_batch in enumerate(training_generator):
            if batch_id - num_skipped > limit:
                # skip to next epoch
                break

            # angle = torch.rand(1) * math.pi
            # rot = torch.tensor([[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]])
            # rotrot = torch.zeros(4, 4)
            # rotrot[:2, :2] = rot.clone()
            # rotrot[2:, 2:] = rot.clone()
            # local_batch[:, :, :, 2:6] = local_batch[:, :, :, 2:6] @ rotrot
            local_batch = local_batch.to(device)
            gt = local_batch.clone()
            model.zero_grad()
            mask, full_peds = get_batch_is_filled_mask(gt)
            if full_peds == 0:
                num_skipped += 1
                continue
            mask = mask.to(device)

            local_batch = local_batch[:, :, :, 2:8]
            if torch.sum(mask) == 0.0:
                num_skipped += 1
                continue
            prediction, kl = model(local_batch[0, :, :, 0:6], train=True)
            gt_prob = torch.cat(([prediction[i].log_prob(gt[0, :, 8 + i, 2:4]) for i in range(12)])).reshape(-1, 12)
            loss_nll = -torch.sum(gt_prob * mask[:, :, 0])
            epoch_loss_nll += loss_nll
            epoch_loss_kl += kl

            loss_stdved = 0.5 * torch.sum(torch.cat(([prediction[i].stddev for i in range(12)])))
            epoch_loss_std += loss_stdved

            loss = loss_nll + 1*loss_stdved - kl

            loss.backward()
            optimizer.step()

            pass
            # plot(predictions, gt)
            epoch_loss += loss.item() / full_peds
            if (batch_id - num_skipped) % 100 == 99:
                print("\tbatch:", str(batch_id - num_skipped), " loss: ", epoch_loss / (batch_id - num_skipped))

        epoch_loss = epoch_loss / (batch_id - num_skipped)
        print("epoch {epoch} loss {el:0.4f}, time taken {t:0.2f}, delta {delta:0.3f}".format(epoch=epoch, el=epoch_loss,
                                                                                             t=time.time() - start,
                                                                                             delta=-prev_epoch_loss + epoch_loss))

        if writer is not None:
            writer.add_scalar(f"loss_epoch", epoch_loss, epoch)
            writer.add_scalar(f"train/kl", epoch_loss_kl, epoch)
            writer.add_scalar(f"train/std", epoch_loss_std, epoch)

        prev_epoch_loss = epoch_loss
        if validate and (epoch % 5 == 0):
            model.eval()
            start = time.time()
            with torch.no_grad():
                ade, fde, nll = get_ade_fde_vel(test_generator, model)
                if min_ade > ade:
                    min_ade = ade
                    torch.save(model.state_dict(), save_model_path)

                if writer is not None:
                    writer.add_scalar(f"test/ade", ade, epoch)
                    writer.add_scalar(f"test/fde", fde, epoch)
                    writer.add_scalar(f"test/nll", nll, epoch)
                t_ade, t_fde, nll = get_ade_fde_vel(training_generator, model, 400)
                if writer is not None:
                    writer.add_scalar(f"train/ade", t_ade, epoch)
                    writer.add_scalar(f"train/fde", t_fde, epoch)
                    writer.add_scalar(f"train/nll", nll, epoch)
            print("\t epoch {epoch} val ade {ade:0.4f}, val fde {fde:0.4f} time taken {t:0.2f}".format(epoch=epoch,
                                                                                                       ade=ade,
                                                                                                       t=time.time() - start,
                                                                                                       fde=fde))
    torch.save(model.state_dict(), save_model_path)


if __name__ == "__main__":
    training_set = Dataset_from_pkl("/home/robot/repos/trajectory-prediction/processed/", data_files=["eth_train.pkl"])
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=1, shuffle=True)

    test_set = Dataset_from_pkl("/home/robot/repos/trajectory-prediction/processed/", data_files=["eth_test.pkl"])
    test_generator = torch.utils.data.DataLoader(test_set, batch_size=1, shuffle=True)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(device)

    model = CvaeFuture(lstm_hidden_dim=32, num_layers=1, bidir=True, dropout_p=0.0, num_modes=30).to(device)
    # model = LstmEncDWithAtt(lstm_hidden_dim=64, num_layers=1,
    #                                              bidir=True, dropout_p=0.0, num_modes=30).to(device)

    train_pose_vel(model, training_generator, test_generator, num_epochs=300, device=device, lr=0.0005, limit=1e600)
