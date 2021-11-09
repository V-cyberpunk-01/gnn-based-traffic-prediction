# -*- coding: utf-8 -*-
"""
@Time   : 2021/4/10

@Author : Chen Shuizhou 祖传读数据
"""
import csv
import torch
import numpy as np
from torch.utils.data import Dataset

# 获得图的邻接矩阵表示
def get_adjacent_matrix(distance_file: str, num_nodes: int, id_file: str = None, graph_type="connect") -> np.array:
    """
    :param distance_file: str, path of csv file to save the distances between nodes.
    :param num_nodes: int, number of nodes in the graph
    :param id_file: str, path of txt file to save the order of the nodes.节点的绝对位置
    :param graph_type: str, ["connect", "distance"] connect表示不考虑节点之间的距离，distance考虑两点之间的距离
    :return:
        np.array(N, N)
    """
    A = np.zeros([int(num_nodes), int(num_nodes)])      # 构造邻接矩阵A

    if id_file:
        with open(id_file, "r") as f_id:
            node_id_dict = {int(node_id): idx for idx, node_id in enumerate(f_id.read().strip().split("\n"))}

            with open(distance_file, "r") as f_d:
                f_d.readline()
                reader = csv.reader(f_d)
                for item in reader:
                    if len(item) != 3:
                        continue
                    i, j, distance = int(item[0]), int(item[1]), float(item[2])
                    if graph_type == "connect":
                        A[node_id_dict[i], node_id_dict[j]] = 1.
                        A[node_id_dict[j], node_id_dict[i]] = 1.
                    elif graph_type == "distance":
                        A[node_id_dict[i], node_id_dict[j]] = 1. / distance
                        A[node_id_dict[j], node_id_dict[i]] = 1. / distance
                    else:
                        raise ValueError("graph type is not correct (connect or distance)")
        return A

    with open(distance_file, "r") as f_d:
        f_d.readline()
        reader = csv.reader(f_d)
        for item in reader:
            if len(item) != 3:
                continue
            i, j, distance = int(item[0]), int(item[1]), float(item[2])  # 两个节点，节点之间的距离

            if graph_type == "connect":         # 两个节点相连
                A[i, j], A[j, i] = 1., 1.
            elif graph_type == "distance":      # 两个节点不相连
                A[i, j] = 1. / distance
                A[j, i] = 1. / distance
            else:
                raise ValueError("graph type is not correct (connect or distance)")
    print(A)
    return A


# 载入流量数据，处理时序数据
def get_flow_data(flow_file: str) -> np.array:
    """
    :param flow_file: str, path of .npz file to save the traffic flow data
    :return:
        np.array(N, T, D) number of node ; time; bias for dim
    """
    data = np.load(flow_file)
    flow_data = data['data'].transpose([1, 0, 2])[:, :, 0][:, :, np.newaxis]  #[N,T，D]

    return flow_data

# 构造自己的数据类型
class LoadData(Dataset):
    def __init__(self, data_path, num_nodes, divide_days, time_interval, history_length, train_mode):
        """
        :param data_path: list, ["graph file name 图结构" , "flow data file name 流量数据"], path to save the data file names.
        :param num_nodes: int, number of nodes.
        :param divide_days: list, [ days of train data , days of test data], 哪几天当数据集，哪几天当测试集
        :param time_interval: int, time interval between two traffic data records (mins).
        :param history_length: int, length time of history data to be used to predict the future
        :param train_mode: list, ["train", "test"].
        """

        self.data_path = data_path
        self.num_nodes = num_nodes
        self.train_mode = train_mode
        self.train_days = divide_days[0]  # 59 -14 = 45
        self.test_days = divide_days[1]  # 14
        self.history_length = history_length  # 6 min
        self.time_interval = time_interval  # 5 min

        self.one_day_length = int(24 * 60 / self.time_interval)

        self.graph = get_adjacent_matrix(distance_file=data_path[0], num_nodes=num_nodes)  #导入图结构

        self.flow_norm, self.flow_data = self.pre_process_data(data=get_flow_data(data_path[1]), norm_dim=1)

    def __len__(self):
        """
        由于目标是一个时间的序列，所以要划分长度
        :return: length of dataset (number of samples for train and test).
        """
        if self.train_mode == "train":
            return self.train_days * self.one_day_length - self.history_length  # 把开头的那一段时间剪出来
        elif self.train_mode == "test":
            return self.test_days * self.one_day_length
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

    def __getitem__(self, index):  # (x, y), index = [0, L1 - 1]
        """
        :param index: int, range between [0, length - 1].
        :return:
            graph: torch.tensor, [N, N].
            data_x: torch.tensor, [N, H, D].
            data_y: torch.tensor, [N, 1, D].
        """
        if self.train_mode == "train":      # 前面的index当作train数据集
            index = index
        elif self.train_mode == "test":     # 后面的部分当作test数据集
            index += self.train_days * self.one_day_length
        else:
            raise ValueError("train mode: [{}] is not defined".format(self.train_mode))

        data_x, data_y = LoadData.slice_data(self.flow_data, self.history_length, index, self.train_mode)

        data_x = LoadData.to_tensor(data_x)  # [N, H, D]  n：number of node; H:history_lenght; D:dim of feature
        data_y = LoadData.to_tensor(data_y).unsqueeze(1)  # [N, 1, D] unsqueeze给y加上一个值为1的维度

        return {"graph": LoadData.to_tensor(self.graph), "flow_x": data_x, "flow_y": data_y}

    @staticmethod
    def slice_data(data, history_length, index, train_mode):
        """
        给时序数据切片，生成input和output
        :param data: np.array, normalized traffic data.
        :param history_length: int, length of history data to be used.
        :param index: int, index on temporal axis.
        :param train_mode: str, ["train", "test"].
        :return:
            data_x: np.array, [N, H, D].
            data_y: np.array [N, D].
        """
        if train_mode == "train":
            start_index = index
            end_index = index + history_length
        elif train_mode == "test":
            start_index = index - history_length
            end_index = index
        else:
            raise ValueError("train model {} is not defined".format(train_mode))

        data_x = data[:, start_index: end_index]
        data_y = data[:, end_index]

        return data_x, data_y

    @staticmethod
    def pre_process_data(data, norm_dim):
        """
        :param data: np.array, original traffic data without normalization.
        :param norm_dim: int, normalization dimension.
        :return:
            norm_base: list, [max_data, min_data], data of normalization base.
            norm_data: np.array, normalized traffic data.
        """
        norm_base = LoadData.normalize_base(data, norm_dim)  # find the normalize base
        norm_data = LoadData.normalize_data(norm_base[0], norm_base[1], data)  # normalize data

        return norm_base, norm_data

    @staticmethod
    def normalize_base(data, norm_dim):
        """
        :param data: np.array, original traffic data without normalization.
        :param norm_dim: int, normalization dimension.
        :return:
            max_data: np.array
            min_data: np.array
        """
        max_data = np.max(data, norm_dim, keepdims=True)  # [N, T, D] , norm_dim=1, [N, 1, D]
        min_data = np.min(data, norm_dim, keepdims=True)
        return max_data, min_data

    @staticmethod
    def normalize_data(max_data, min_data, data):
        """
        :param max_data: np.array, max data.
        :param min_data: np.array, min data.
        :param data: np.array, original traffic data without normalization.
        :return:
            np.array, normalized traffic data.
        """
        mid = min_data
        base = max_data - min_data
        normalized_data = (data - mid) / base

        return normalized_data

    @staticmethod
    def recover_data(max_data, min_data, data):
        """
        :param max_data: np.array, max data.
        :param min_data: np.array, min data.
        :param data: np.array, normalized data.
        :return:
            recovered_data: np.array, recovered data.
        """
        mid = min_data
        base = max_data - min_data

        recovered_data = data * base + mid

        return recovered_data

    @staticmethod
    def to_tensor(data):
        return torch.tensor(data, dtype=torch.float)


if __name__ == '__main__':
    train_data = LoadData(data_path=["PeMS_04/PeMS04.csv", "PeMS_04/PeMS04.npz"], num_nodes=307, divide_days=[45, 14],# 前45天做测试集
                          time_interval=5 , history_length=6 ,  # 数据每5分钟有一个记录
                          train_mode="train")

    print(len(train_data))
    print(train_data[0]["flow_x"].size())
    print(train_data[0]["flow_y"].size())
