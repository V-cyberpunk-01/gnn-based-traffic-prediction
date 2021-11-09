# -*- coding: utf-8 -*-
"""
@Time   : 2021/4/10

@Author : Chen Shuizhou
"""
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from traffic_dataset import LoadData
from chebnet import ChebNet
import matplotlib.pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"

class GCN(nn.Module):
    def __init__(self, in_c, hid_c, out_c):
        super(GCN, self).__init__()
        self.linear_1 = nn.Linear(in_c, hid_c)
        self.linear_2 = nn.Linear(hid_c, out_c)
        self.act = nn.ReLU()

    def forward(self, data, device):
        graph_data = data["graph"].to(device)[0]  # [N, N]
        graph_data = GCN.process_graph(graph_data)

        flow_x = data["flow_x"].to(device)  # [B, N, H, D]

        B, N = flow_x.size(0), flow_x.size(1)

        flow_x = flow_x.view(B, N, -1)  # [B, N, H*D]  H = 6, D = 1 维度压缩

        output_1 = self.linear_1(flow_x)  # [B, N, hid_C]
        output_1 = self.act(torch.matmul(graph_data, output_1))  # [N, N], [B, N, Hid_C]

        output_2 = self.linear_2(output_1)
        output_2 = self.act(torch.matmul(graph_data, output_2))  # [B, N, 1, Out_C]

        return output_2.unsqueeze(2)  # 加上时间维度

    @staticmethod
    def process_graph(graph_data):
        N = graph_data.size(0)  # 获得图数据库的节点数量
        matrix_i = torch.eye(N, dtype=graph_data.dtype, device=graph_data.device)  # 创建单位矩阵，数据类型和graphdata的类型一样
        graph_data += matrix_i  # A~ [N, N]

        degree_matrix = torch.sum(graph_data, dim=-1, keepdim=False)  # [N]
        degree_matrix = degree_matrix.pow(-1)   # [N]的逆
        degree_matrix[degree_matrix == float("inf")] = 0.  # [N]

        degree_matrix = torch.diag(degree_matrix)  # [N, N]

        return torch.mm(degree_matrix, graph_data)  # D^(-1) * A = \hat(A)

# 用及其简单的baseline测试一下整体的框架是不是对的
class Baseline(nn.Module):
    def __init__(self, in_c, out_c):
        super(Baseline, self).__init__()
        self.layer = nn.Linear(in_c, out_c)

    def forward(self, data, device):
        flow_x = data["flow_x"].to(device)  # [B, N, H, D]
        B, N = flow_x.size(0), flow_x.size(1)
        flow_x = flow_x.view(B, N, -1)  # [B, N, H*D]  H = 6, D = 1
        output = self.layer(flow_x)  # [B, N, Out_C],+ Out_C = D

        return output.unsqueeze(2)  # [B, N, 1, D=Out_C]


def main():


    # Loading Dataset
    train_data = LoadData(data_path=["PeMS_04/PeMS04.csv", "PeMS_04/PeMS04.npz"], num_nodes=307, divide_days=[45, 14],
                          time_interval=5, history_length=6,
                          train_mode="train")

    train_loader = DataLoader(train_data, batch_size=64, shuffle=True, num_workers=8)

    test_data = LoadData(data_path=["PeMS_04/PeMS04.csv", "PeMS_04/PeMS04.npz"], num_nodes=307, divide_days=[45, 14],
                         time_interval=5, history_length=6,
                         train_mode="test")

    test_loader = DataLoader(test_data, batch_size=64, shuffle=True, num_workers=8)

    # Loading Model
    # my_net = ChebNet(in_c = 6, hid_c= 6, out_c = 1, K = 2)
    # my_net = Baseline(in_c=6, out_c=1)
    my_net = GCN(in_c= 6,hid_c= 6,out_c= 1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    my_net = my_net.to(device)

    criterion = nn.MSELoss() # 定义lossfunction是MSELoss

    optimizer = optim.Adam(params=my_net.parameters())

    # Train model
    Epoch = 10

    my_net.train()
    for epoch in range(Epoch):
        epoch_loss = 0.0
        start_time = time.time()
        for data in train_loader:  # ["graph": [B, N, N] , "flow_x": [B, N, H, D], "flow_y": [B, N, 1, D]]
            my_net.zero_grad()  # 每一次的梯度都给他清0
            predict_value = my_net(data, device).to(torch.device("cpu"))  # [0, 1] -> recover到cpu上
            loss = criterion(predict_value, data["flow_y"])
            epoch_loss += loss.item()  # 损失函数的loss的值拿出来

            loss.backward()

            optimizer.step()
        end_time = time.time()

        print("Epoch: {:04d}, Loss: {:02.4f}, Time: {:02.2f} mins".format(epoch, 1000 * epoch_loss / len(train_data),
                                                                          (end_time-start_time)/60))

    # Test Model
    # TODO: Visualize the Prediction Result
    # TODO: Measure the results with metrics MAE, MAPE, and RMSE
    my_net.eval()  #关闭训练模式打开测试模式
    with torch.no_grad():
        total_loss = 0.0
        for data in test_loader:
            predict_value = my_net(data, device).to(torch.device("cpu"))  # [B, N, 1, D] 把图数据也转入gpu中
            print(predict_value[3])
            loss = criterion(predict_value, data["flow_y"])
            total_loss += loss.item()
        print("Test Loss: {:02.4f}".format(1000 * total_loss / len(test_data)))

    # make prediction




if __name__ == '__main__':
    main()
