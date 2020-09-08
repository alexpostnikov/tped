from torch.utils.data import Dataset
import os
import torch
import dill
from typing import Union, List, Set
from tqdm import tqdm


def find_next_data_by_index(index, dataset):

    self_data = dataset[index]
    last_seen_timestamp = dataset[index][1].item()
    end_ts = last_seen_timestamp + 20
    person_id = self_data[0].item()
    n_history_tensor = torch.zeros(20, 14)
    history_data = dataset[
        torch.where(
            (dataset[:, 1] >= last_seen_timestamp) * (dataset[:, 1] < end_ts) * (dataset[:, 0] == person_id))]
    last_seen_timestamp = history_data[-1, 1].item()
    n_history_tensor[0:len(history_data), :] = history_data

    return n_history_tensor, last_seen_timestamp


def get_neighbours_history(neighbours_indexes, data, start_timestamp):

    end_timestamp = start_timestamp + 8
    n_history_tensor = torch.zeros(len(neighbours_indexes), 8, 14)

    for i, person_id in enumerate(neighbours_indexes):
        neighbour = data[
            torch.where((data[:, 1] >= start_timestamp) * (data[:, 1] < end_timestamp) * (data[:, 0] == person_id))]
        if len(neighbour) > 0:
            num_missed_ts_from_start = int(neighbour[0, 1] - start_timestamp)
            n_history_tensor[i][num_missed_ts_from_start:num_missed_ts_from_start + len(neighbour)] = neighbour
    return n_history_tensor


def get_peds(index, dataset: List):
    """
    return stacked torch tensor of scene in specified timestamps from specified dataset.
    if at any given timestamp person is not found at dataset, but later (previously) will appear,
    that its tensor data is tensor of zeros.
    :param index:

    :param dataset: list of data. shapes are: 1: timestamp 1: num_peds, 2: RFU, 3: data np.array of 8 floats
    :return: torch tensor of shape : end-start, max_num_peds, , 20 , 8
    """
    self_data = dataset[index]
    start = int(self_data[1].item())
    self_index = self_data[0].item()

    person_history, end = find_next_data_by_index(index, dataset)
    end = start + 8
    # end = int(person_history[-1, 1].item())
    neighbours_indexes = set()
    counter = 0

    while index - counter >= 0:
        neighbours_indexes.add(dataset[index - counter][0].item())
        counter += 1
        if dataset[index - counter][1] != start:
            break

    counter = 0
    while dataset[index + counter][1] < end:
        neighbours_indexes.add(dataset[index + counter][0].item())
        counter += 1
        if index + counter >= len(dataset):
            break
    pass
    neighbours_indexes.remove(self_index)
    neighbours_history = get_neighbours_history(list(neighbours_indexes), dataset,
                                                start)

    return person_history, neighbours_history


def get_peds_indexes_in_timestamp(person: List):
    """
    :param person:
    :return: Set of  peds ids at scene
    """
    indexes = []

    if type(person) == torch.Tensor:
        indexes.append(float(person[0]))
    else:
        indexes.append(float(person[0][0]))
    return set(indexes)


def get_peds_indexes_in_range_timestamps(start: int, end: int, dataset: List) -> Set:
    """
    :param start: timestamp start
    :param end:  timestamp end
    :param dataset: list of data. shapes are: 0: timestamp 1: num_peds, 2: RFU, 3: data np.array of 8 floats
    :return: dict of  predestrians ids at each scene (one scene is 20 timestamps)
    """
    indexes = []
    for time_start in range(start, end):

        for duration in range(0, 20):
            peoples = dataset[time_start + duration]
            indexes += list(get_peds_indexes_in_timestamp(peoples))

    return set(indexes)


class DatasetFromPkl(Dataset):
    """
        Class for loading data in torch format from preprocessed pkl
        files with pedestrian poses, velocities and accelerations
    """

    def __init__(self, data_folder: str, data_files: Union[str, List[str]] = "all",
                 train: bool = True, test: bool = False, validate: bool = False):
        """
        :param data_folder: path to folder with preprocessed pkl files
        :param data_files: list of files to be used or "all"
        :param train: if data_files is all, if train is false -> all *train*.pkl files will be ignored
        :param test: if data_files is all, if train is false -> all *test*.pkl files will be ignored
        :param validate: if data_files is all, if train is false -> all *val*.pkl files will be ignored
        """

        super().__init__()
        self.train_dataset = torch.tensor([])
        file_list = []
        if "all" not in data_files:
            file_list = data_files
        else:
            dd = os.listdir(data_folder)
            for file in dd:
                if train and "train" in file:
                    file_list.append(file)
                if test and "test" in file:
                    file_list.append(file)
                if validate and "val" in file:
                    file_list.append(file)
        data_dict = {}
        for x in file_list:
            data_dict[x] = data_folder + "/" + x
        self.data = {}
        for file_name in data_dict.keys():
            with open(data_dict[file_name], 'rb') as f:
                print("loading " + file_name)
                self.data[file_name] = dill.load(f)
                for j in range(len(self.data[file_name])):
                    self.data[file_name][j] = torch.cat(self.data[file_name][j])

        self.dataset_indeces = {}
        self.data_length = 0

        for key in self.data.keys():
            for index, sub_dataset in enumerate(self.data[key]):
                self.data_length += len(sub_dataset) - 20
                self.dataset_indeces[self.data_length] = [key, index]
        data_dim = sub_dataset.shape[-1]

        self.processed_history = torch.zeros(self.data_length, 20, data_dim)
        self.processed_neighbors = [torch.tensor([]) for _ in range(self.data_length)]

        self.upper_bounds = list(self.dataset_indeces.keys())
        self.upper_bounds.append(0)
        self.upper_bounds.sort()

    def limit_len(self, new_len):
        self.data_length = new_len

    def get_dataset_from_index(self, data_index: int):
        """
        given index return dataset name and sub_dataset id, corresponding index in sub_dataset
        :param data_index: data sample number
        :return: file_name, sub_dataset id, corresponding index in sub_dataset
        """
        upper_bounds = list(self.dataset_indeces.keys())
        upper_bounds.append(0)
        upper_bounds.sort()
        index = [upper_bound > data_index for upper_bound in upper_bounds].index(True)
        index_in_sub_dataset = data_index - upper_bounds[index - 1]
        return self.dataset_indeces[upper_bounds[index]], index_in_sub_dataset

    def __len__(self):
        return self.data_length

    def load_from_preprocessed(self, file, sub_dataset, index_in_sub_dataset):

        node_hist = torch.load("preprocessed/"
                               + file[:file.index(".")] + str(sub_dataset) + ".pt")
        neig_hist = torch.load("preprocessed/"
                               + file[:file.index(".")] + str(sub_dataset) + "neighbors.pt")

        for i, (dataset_name, dataset_index) in enumerate(self.dataset_indeces.values()):
            if dataset_name == file and dataset_index == sub_dataset:
                self.processed_history[self.upper_bounds[i]:self.upper_bounds[i + 1]] = node_hist
                self.processed_neighbors[self.upper_bounds[i]:self.upper_bounds[i + 1]] = neig_hist
        return [node_hist[index_in_sub_dataset], neig_hist[index_in_sub_dataset]]

    def __getitem__(self, index: int):
        [file, sub_dataset], index_in_sub_dataset = self.get_dataset_from_index(index)

        if not torch.any(self.processed_history[index] != torch.zeros_like(self.processed_history[index])):

            if os.path.isfile("preprocessed/"
                              + file[:file.index(".")] + str(sub_dataset) + ".pt"):
                data = self.load_from_preprocessed(file, sub_dataset, index_in_sub_dataset)
                return data

            self.packed_data = []  # num_samples * 20 * numped * 8
            data = get_peds(index_in_sub_dataset, self.data[file][sub_dataset])
            self.processed_history[index] = data[0]
            self.processed_neighbors[index] = data[1]
            return data
        else:
            data = [self.processed_history[index], self.processed_neighbors[index]]
            return data

    def save_preprocessed(self):
        for i in tqdm(range(self.data_length)):
            self.__getitem__(i)
        upper_bounds = list(self.dataset_indeces.keys())
        upper_bounds.append(0)
        upper_bounds.sort()
        for i, (dataset_name, index) in enumerate(self.dataset_indeces.values()):
            torch.save(self.processed_history[upper_bounds[i]:upper_bounds[i + 1]], "preprocessed/"
                       + dataset_name[:dataset_name.index(".")] + str(index) + ".pt")
            torch.save(self.processed_neighbors[upper_bounds[i]:upper_bounds[i + 1]], "preprocessed/"
                       + dataset_name[:dataset_name.index(".")] + str(index) + "neighbors.pt")


def collate_fn(data):
    node_hist = []
    neighbours = []
    for i in range(len(data)):
        node_hist.append(data[i][0])
        neighbours.append(data[i][1][:, :8, 2:8])
    node_hist = torch.stack(node_hist)
    return node_hist, neighbours


def is_filled(data):
    return not (data[:, 1] == 0).any().item()


if __name__ == "__main__":
    pass
    # dataset = DatasetFromPkl("/home/robot/repos/trajectory-prediction/processed_with_forces/",
    #                          data_files=["eth_train.pkl"])  # , "zara2_test.pkl"]
    # # dataset.save_preprocessed()
    # print(len(dataset))
    # t = dataset[0]
    # print(dataset[0][0].shape)
    #
    # # training_set = Dataset_from_pkl("/home/robot/repos/trajectories_pred/processed/", data_files=["eth_train.pkl"])
    # training_generator = torch.utils.data.DataLoader(dataset, batch_size=512, collate_fn=collate_fn)  # , num_workers=10
    # #
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
