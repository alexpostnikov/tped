import torch
from dataloader_parallel import DatasetFromPkl, collate_fn, is_filled
from model.model_cvae_parallel import CvaeFuture


from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import numpy as np
import io
import time
import cv2

import matplotlib.pyplot as plt

def plot_prob(gt, gmm, ped_num):
    fig, ax = plt.subplots(4, 3, figsize=(18, 18))
    for i in range(12):
        plot_prob_step(gt[8+i], gmm[i], ped_num, ax[i % 4][i // 4])
        ax[i % 4][i // 4].set_title("timestamp " + str(i))

    fig.show()
    pass


counter = 0

def plot_prob_big(gt, gmm, ped_num, save=True, device="cpu", add_map=False,timestamp = None):
    global counter
    # k=24
    xp = torch.arange(torch.min(gt[8:, 0]) - 2, torch.max(gt[8:, 0]) + 2, 0.05).to(device)
    yp = torch.arange(torch.min(gt[8:, 1]) - 2, torch.max(gt[8:, 1]) + 2, 0.05).to(device)
    if add_map:
        video = cv2.VideoCapture("./resources/seq_eth.avi")
        if timestamp is not None:
            video.set(cv2.CAP_PROP_POS_FRAMES, timestamp[0,0])
        ret, frame = video.read()
        H = np.array([[2.8128700e-02,   2.0091900e-03,  -4.6693600e+00],
                    [8.0625700e-04,   2.5195500e-02,  -5.0608800e+00],
                    [3.4555400e-04,   9.2512200e-05,   4.6255300e-01]])
        Hi = np.linalg.inv(H)
        k=28
        x = torch.arange(0., frame.shape[0]).to(device)
        y = torch.arange(0., frame.shape[1]).to(device)
    gt = gt.to(device)
    prob = torch.zeros_like(yp[None].T * xp).to(device)
    for t in range(12):
        X1 = xp.unsqueeze(0)
        Y1 = yp.unsqueeze(1)
        X2 = X1.repeat(yp.shape[0], 1)
        Y2 = Y1.repeat(1, xp.shape[0])
        Z = torch.stack([X2, Y2]).permute(1, 2, 0)
        Z = Z.reshape(-1, 2)
        ll = gmm[t].log_prob(Z.unsqueeze(1))[:, ped_num]
        prob += torch.exp(ll).reshape(yp.shape[0], xp.shape[0])
    prob = torch.clamp(prob, max=2)
    # gtn = np.array(gt)
    # for i in range(len(gtn)):
    #     gtn[i] = (Hi@np.array([gtn[i][1],gtn[i][0],1]))[:2]
    # temp = gtn.copy()
    # gtn[:, 0] =  640-temp[:, 0]
    if add_map:
        temp = gt.clone()
        gt[:, 0] =  temp[:, 1]
        gt[:, 1] =  temp[:, 0]
        
        gt[:, 0] = (gt[:, 0] + frame.shape[1]/k/2)*k
        gt[:, 1] = (gt[:, 1] + frame.shape[0]/k/2)*k
    else:
        gt[:, 0] = (gt[:, 0] - xp.mean())/(torch.max(xp) - torch.min(xp)) * (len(xp)) + (len(xp)/2.0)
        gt[:, 1] = (gt[:, 1] - yp.mean())/(torch.max(yp) - torch.min(yp)) * (len(yp)) + (len(yp)/2.0)
    fig, ax = plt.subplots(1, figsize=(18, 18))
    ax.set_xticks(np.round(np.linspace(0, len(xp), 6), 1))
    ax.set_yticks(np.round(np.linspace(0, len(yp), 6), 1))
    ax.set_xticklabels(
        np.round(((torch.linspace(0, len(xp), 6) * (torch.max(xp) - torch.min(xp)) / len(xp) + torch.min(xp)).numpy()), 1))
    ax.set_yticklabels(
        np.round((((torch.linspace(0, len(xp), 6)) * (torch.max(xp) - torch.min(xp)) / len(yp) + torch.min(yp)).numpy()), 1))
    # ax.plot(gtn[:, 0], gtn[:, 1])
    # ax.plot(gtn[:, 0], gtn[:, 1], "bo")
    # ax.plot(gtn[8:, 0], gtn[8:, 1], 'ro')
    ax.plot(gt[:, 0], gt[:, 1])
    ax.plot(gt[:, 0], gt[:, 1], "b*")
    ax.plot(gt[8:, 0], gt[8:, 1], "r*")
    p1 = [9999,9999]
    p2 = [0,0]
    if add_map:
        for i in range(len(xp)):
            for j in  range(len(yp)):
                try:
                    xf = int((xp[i] + frame.shape[0]/k/2)*k)
                    yf = int((yp[j] + frame.shape[1]/k/2)*k)
                    xf = max([xf,0])
                    xf = min([xf,frame.shape[0]])
                    yf = max([yf,0])
                    yf = min([yf,frame.shape[1]])
                    p1[0]=min([xf,p1[0]])
                    p1[1]=min([yf,p1[1]])
                    p2[0]=max([xf,p2[0]])
                    p2[1]=max([yf,p2[1]])
                    # xf = int((Hi@np.array([xp[i][1],xp[i][0],1]))[:2])
                    # yf = int((Hi@np.array([yp[j][1],yp[j][0],1]))[:2])
                    # yf = int((yp[j] + frame.shape[1]/k/2)*k)
                    
                    frame[xf][yf][0] = prob[j][i]*127#(max 2 to 254)
                    frame[xf][yf][1] *= 0.1 #G
                    frame[xf][yf][2] *= 0.1 #B
                except:
                    pass
        frame[p1[0]:p2[0],p1[1]:p2[1]] = cv2.blur(frame[p1[0]:p2[0],p1[1]:p2[1]],(10,10))
        ax.imshow(frame)
    else:
        ax.imshow(prob.detach().cpu())
    if save:
        plt.savefig("./visualisations/traj_dirtr/"+str(counter)+".jpg", )
        plt.close()
    counter += 1

    return ax


def plot_prob_step(gt, gmm, ped_num, ax):
        x = torch.arange(gt[0] - 2, gt[0] + 2, 0.1).to(gt.device)
        y = torch.arange(gt[1] - 2, gt[1] + 2, 0.1).to(gt.device)
        prob = torch.zeros_like(x[None].T * y)
        for i in range(len(x)):
            for j in range(len(y)):
                prob[i][j] = torch.exp(gmm.log_prob(torch.tensor([x[i], y[j]]).to(gt.device))[ped_num])
        prob = prob.clamp(max=2)
        ax.imshow(prob.detach().cpu())

        # ax.set_xticks(np.arange(len(labels_x)))
        # ax.set_yticks(np.arange(len(labels_y)))

        # ax.set_xticklabels(labels_x.numpy())
        # ax.set_yticklabels(labels_y.numpy())
        ax.set_xticks(np.arange(0, len(x), 6))
        ax.set_yticks(np.arange(0, len(y), 6))
        ax.set_xticklabels(np.round((((torch.arange(0, len(x), 6))*(torch.max(x) - torch.min(x)) / len(x) + torch.min(x)).numpy()),1))
        ax.set_yticklabels(np.round(((( torch.arange(0, len(y), 6))*(torch.max(x) - torch.min(x)) / len(y) + torch.min(y)).numpy()),1))
        # for _ in range(20):
        #     mean_pr = gmm.sample()
        #     # mean_pr = torch.tensor([5.,5])
        #
        #     mean_pr[0] = (-torch.min(x) + mean_pr[0]) * len(x) / (torch.max(x) - torch.min(x))
        #     mean_pr[1] = (-torch.min(x) + mean_pr[1]) * len(x) / (torch.max(x) - torch.min(x))
        #
        #     circle = plt.Circle((mean_pr[0], mean_pr[1]), radius=0.4, color="r")
        #     ax.add_patch(circle)
        mean_pr = torch.zeros(2)
        mean_pr[0] = (-torch.min(x) + gt[0]) * len(x) / (torch.max(x) - torch.min(x))
        mean_pr[1] = (-torch.min(y) + gt[1]) * len(x) / (torch.max(y) - torch.min(y))

        circle = plt.Circle((mean_pr[0], mean_pr[1]), radius=0.4, color="r")
        ax.add_patch(circle)
        return ax


def plot_traj(data, ax=None, color="red"):
    # data shape [n, 2]
    x_poses = np.zeros((data.shape[0], data.shape[1]))
    y_poses = np.zeros((data.shape[0], data.shape[1]))
    for person in range(data.shape[0]):
        x_poses[person] = data[person, :, 0].numpy()
        y_poses[person] = data[person, :, 1].numpy()

    if data.shape[0] < 2:
        for person in range(data.shape[0]):
            if ax is not None:
                ax.plot(x_poses[person], y_poses[person], 'o-', color=color)
            else:
                plt.plot(x_poses[person], y_poses[person], 'o-', color=color)
                plt.show()
    else:
        for person, color_index in enumerate(np.linspace(0, 1, data.shape[0])):
            if ax is not None:
                ax.plot(x_poses[int(person)], y_poses[int(person)], 'o-', color=plt.cm.RdYlBu(color_index))
            else:
                plt.plot(x_poses[int(person)], y_poses[int(person)], 'o-', color=plt.cm.RdYlBu(color_index))
                plt.show()


def visualize(model, gen, limit=10e7, device="cuda"):
    for batch_id, local_batch in enumerate(gen):
        local_batch = local_batch.to(device)
        if local_batch.shape[1] < 1:
            continue
        if batch_id > limit:
            # stop to next epoch
            break
        gt = local_batch.clone()
        # local_batch = local_batch[:, :, :8, 2:4].to(device)
        # local_batch[0, :, 8:, :] = torch.zeros_like(local_batch[0, :, 8:, :]).to(device)

        num_peds = local_batch.shape[1]
        # predictions = torch.zeros(num_peds, 0, 2).requires_grad_(True).to(device)
        prediction = model(local_batch[0, :, :, 2:8])
        predictions = torch.cat([prediction[i].mean for i in range(12)]).reshape(12, -1, 2).permute(1, 0, 2)
        for ped_num in range(num_peds):
            if is_filled(local_batch[0, ped_num, :8, :]):

                if not torch.any(torch.norm(gt[0, ped_num, 8:, 2:4],dim=-1)==torch.tensor([0]).to(device)):
                    ax = plot_prob_big(gt[0, ped_num, :, 2:4], prediction, ped_num, device=device)
                    return ax
                    # plot_prob(gt[0, ped_num, :, 2:4], prediction, ped_num)

                # fig = plt.figure()
                # ax1 = fig.add_subplot(2, 1, 1)
                # ax2 = fig.add_subplot(2, 1, 2, sharex=ax1)
                # # fig, ax = plt.subplots(1, 3, figsize=(9, 3))
                # plot_traj(local_batch[0, :, :8, 2:4].detach().cpu(), ax=ax1, color="blue")
                #
                # # data = local_batch[0, :, :, 2:4].cpu()
                # # plot_traj(data, ax=ax[0], color="blue")
                #
                # # data = torch.cat((gt[0,ped_num:ped_num+1,0:8,2:4], predictions[ped_num:ped_num+1,:,:].detach().cpu()),dim=1)\
                #
                # # plot_traj(gt[0, ped_num:ped_num + 1, 0:8, 2:4], ax[2], color="blue")
                #
                # plot_traj(predictions[ped_num:ped_num + 1, :, :].detach().cpu(), ax2)
                # plot_traj(gt[0, ped_num:ped_num + 1, 8:, 2:4].detach().cpu(), ax2, color="black")
                # plot_traj(local_batch[0, ped_num:ped_num + 1, :8, 2:4].detach().cpu(), ax2, color="blue")
                # pass
                # plt.show()


def visualize_single(model, gen, device="cuda"):
    for batch_id, local_batch in enumerate(gen):
        local_batch = local_batch.to(device)
        if local_batch.shape[1] < 1:
            continue
        gt = local_batch.clone()
        # local_batch = local_batch[:, :, :8, 2:4].to(device)
        # local_batch[0, :, 8:, :] = torch.zeros_like(local_batch[0, :, 8:, :]).to(device)

        num_peds = local_batch.shape[1]
        # predictions = torch.zeros(num_peds, 0, 2).requires_grad_(True).to(device)
        prediction = model(local_batch[0, :, :, 2:8])
        predictions = torch.cat([prediction[i].mean for i in range(12)]).reshape(12, -1, 2).permute(1, 0, 2)
        for ped_num in range(num_peds):
            if is_filled(local_batch[0, ped_num, :8, :]):

                if not torch.any(torch.norm(gt[0, ped_num, 8:, 2:4],dim=-1) == torch.tensor([0]).to(device)):
                    ax = plot_prob_big(gt[0, ped_num, :, 2:4], prediction, ped_num, device=device)
                    return ax




def visualize_single_parallel(model, gen, device="cpu"):
    for batch_id, local_batch in enumerate(gen):

        x, neighbours = local_batch
        x = x.to(device)
        gt = x.clone()
        model.to(device)
        model.zero_grad()

        x = x[:, :, 2:8]
        prediction = model(x[:, :, 0:6], neighbours, train=False)
        gt_prob = torch.cat(([prediction[i].log_prob(gt[:, 8 + i, 2:4]) for i in range(12)])).reshape(-1, 12)
        num_peds = x.shape[0]
        # predictions = torch.cat([prediction[i].mean for i in range(12)]).reshape(12, -1, 2).permute(1, 0, 2)
        for ped_num in range(num_peds):
            if is_filled(x[ped_num, :8, :]):

                if not torch.any(torch.norm(gt[ped_num, 8:, 2:4], dim=-1) == torch.tensor([0]).to(device)):
                    ax = plot_prob_big(gt[ped_num, :, 2:4], prediction, ped_num, device=device,add_map=False ,timestamp = gt[ped_num, :, 0:2])
                    return ax



if __name__ == "__main__":

    training_set = DatasetFromPkl("resources/processed_with_forces/",
                                  data_files=["eth_train.pkl"])
    training_generator = torch.utils.data.DataLoader(training_set, batch_size=2, collate_fn=collate_fn, shuffle=True)

    test_set = DatasetFromPkl("resources/processed_with_forces/",
                              data_files=["eth_test.pkl"])
    test_generator = torch.utils.data.DataLoader(test_set, batch_size=12, shuffle=True, collate_fn=collate_fn)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    print(device)
    model = CvaeFuture(lstm_hidden_dim=64, num_layers=1, bidir=True, dropout_p=0.0, num_modes=30).to(device)
    model.load_state_dict(torch.load(
        "resources/model.pth",map_location=torch.device('cpu')))


    start = time.time()
    for i in range(2):
        ax = visualize_single_parallel(model, test_generator)
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)

    print(time.time() - start)