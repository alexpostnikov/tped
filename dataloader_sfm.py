from torch.utils.data import Dataset
import os
import torch
import dill
from typing import Union, List, Set
from tqdm import tqdm
from dataloader_parallel import DatasetFromPkl


class DataloaderSfm(Dataset):
    def __init__(self, data_folder: str, data_files: Union[str, List[str]], sfm_file: str):

        super().__init__()
        self.dataset = DatasetFromPkl(data_folder,
                                 data_files)
        self.sfm_t = torch.load(data_folder+sfm_file)
        self.data_length = len(self.dataset)-1
        # print(len(self.dataset))
        # print(len(self.sfm_t))

    def __len__(self):
        return self.data_length

    def __getitem__(self, index: int):
        data = self.dataset[index]
        sfm_data = self.sfm_t[index]
        data.append(sfm_data)
        return data


def collate_fn(data):
    node_hist = []
    neighbours = []
    sfm_pred = []
    for i in range(len(data)):
        node_hist.append(data[i][0])
        neighbours.append(data[i][1][:, :8, 2:8])
        sfm_pred.append(data[i][2])

    node_hist = torch.stack(node_hist)
    sfm_pred = torch.stack(sfm_pred)
    return node_hist, neighbours, sfm_pred


if __name__ == "__main__":
    dataset = DataloaderSfm("/home/robot/repos/tped/processed_with_forces/",
                             data_files=["eth_train.pkl"], sfm_file="sfm.pkl")

    training_generator = torch.utils.data.DataLoader(dataset, batch_size=256, collate_fn=collate_fn)  # , num_workers=10
    # for i in tqdm(range(len(dataset))):
    #     a = dataset[i]
    for i in tqdm(training_generator):
        pass
    print("done")

    # # dataset.save_preprocessed()
    # print(len(dataset))
    # t = dataset[0]
    # print(dataset[0][0].shape)
    # # training_set = Dataset_from_pkl("/home/robot/repos/trajectories_pred/processed/", data_files=["eth_train.pkl"])
    # training_generator = torch.utils.data.DataLoader(dataset, batch_size=512, collate_fn=collate_fn# , num_workers=10
    # import time
    #
    # start = time.time()
    # for i, local_batch in enumerate(training_generator):
    #     if i > 5:
    #         break
    #     print(i)
    #     # print(local_batch[0].shape)
    #     # print(local_batch[0][-1, 0, 1])
    #     pass
    #
    # print(time.time() - start)
    #
    # start = time.time()
    # for i, local_batch in enumerate(training_generator):
    #     if i > 5:
    #         break
    #     print(i)
    #
    #     pass
    #
    # print(time.time() - start)
