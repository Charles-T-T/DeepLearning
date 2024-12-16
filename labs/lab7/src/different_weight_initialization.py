import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import time
import torch.optim as optim

"""
本程序用于简单对比一下不同的权重初始化方式
"""


# 参数
LEARNING_RATE = 0.001
MAX_EPOCH = 5
BATCH_SIZE = 64


def generate_dataset(dataset_path="./mnist/"):
    """
    用于获取数据集
    """
    # 下载MNIST数据集
    DOWNLOAD_MNIST = False
    if not (os.path.exists(dataset_path)) or not os.listdir(dataset_path):
        DOWNLOAD_MNIST = True

    # 设置训练集
    train_data = torchvision.datasets.MNIST(
        root=dataset_path,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=DOWNLOAD_MNIST,
    )
    train_loader = Data.DataLoader(
        dataset=train_data, batch_size=BATCH_SIZE, shuffle=True
    )

    # 设置测试集
    test_data = torchvision.datasets.MNIST(root="./mnist/", train=False)
    test_x = (
        torch.unsqueeze(test_data.data, dim=1).type(torch.FloatTensor)[:500] / 255.0
    )
    test_y = test_data.test_labels[:500]
    return train_loader, test_x, test_y


class Net_normal_init(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=100, output_dim=10):
        super(Net_normal_init, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # 对线性层权重正态分布初始化
        nn.init.normal_(self.fc1.weight, mean=0, std=1)
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # 对线性层权重正态分布初始化
        nn.init.normal_(self.fc2.weight, mean=0, std=1)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x


class Net_uniform_init(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=100, output_dim=10):
        super(Net_uniform_init, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        ###########################################
        # 填空(可参考上面的例子)
        # 对线性层权重(fc1)均匀分布初始化(用nn.init.uniform_)
        nn.init.uniform_(self.fc1.weight)
        ###########################################
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        ###########################################
        # 填空(可参考上面的例子)
        #  对线性层权重(fc2)均匀分布初始化(用nn.init.uniform_)
        nn.init.uniform_(self.fc2.weight)
        ###########################################

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x


class Net_Xavier_init(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=100, output_dim=10):
        super(Net_Xavier_init, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        ################################################
        # 填空(可参考上面的例子)
        # 对线性层权重(fc1)使用Xavier初始化
        nn.init.xavier_uniform_(self.fc1.weight)
        ################################################
        self.sigmoid = nn.Sigmoid()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        ###################################################
        # 填空
        # 对线性层权重(fc2)使用Xavier初始化
        nn.init.xavier_uniform_(self.fc2.weight)
        ###################################################

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.sigmoid(x)
        x = self.fc2(x)
        return x


# 测试
def test(model, test_x, test_y):
    model.eval()
    y_pre = model(test_x)
    _, pre_index = torch.max(y_pre, 1)
    pre_index = pre_index.view(-1)
    prediction = pre_index.numpy()
    correct = np.sum(prediction == test_y.numpy())
    return correct / 500.0


# 训练
def train(model, optimizer, criterion, train_loader, test_x, test_y):
    train_loss_list = []
    test_loss_list = []
    iteration_num_list = []
    precision_list = []

    iteration_num = 0
    for epoch in range(MAX_EPOCH):
        start_time = time.time()
        for step, (x, y) in enumerate(train_loader):
            # 进行每一个iteration训练
            model.train()
            output = model(x)
            train_loss = criterion(output, y)
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()

            if step % 20 == 0:  # 每20轮进行记录
                model.eval()
                with torch.no_grad():
                    iteration_num += 20
                    precision = test(model, test_x, test_y)
                    test_loss = criterion(model(test_x), test_y)
                    test_loss_list.append(test_loss.item())
                    train_loss_list.append(train_loss.item())
                    precision_list.append(precision)
                    iteration_num_list.append(iteration_num)
        print(
            "epoch: {}, train_loss {:.3f}, test_loss {:.3f} precision {:.3f} using time {:.3f}".format(
                epoch,
                train_loss_list[-1],
                test_loss_list[-1],
                precision_list[-1],
                time.time() - start_time,
            )
        )
    return train_loss_list, test_loss_list, precision_list, iteration_num_list


if __name__ == "__main__":
    fig = plt.figure()
    pre_ax = plt.subplot(131)  # 用于可视化准确率变化
    train_loss_ax = plt.subplot(132)  # 用于可视化训练损失变化
    test_loss_ax = plt.subplot(133)  # 用于可视化测试损失变化

    # 设置可视化图像标题与坐标轴标签
    pre_ax.set_title("precision")
    pre_ax.set_xlabel("iter")
    pre_ax.set_ylabel("pre")
    train_loss_ax.set_title("train_loss")
    train_loss_ax.set_xlabel("iter")
    train_loss_ax.set_ylabel("loss")
    test_loss_ax.set_title("test_loss")
    test_loss_ax.set_xlabel("iter")
    test_loss_ax.set_ylabel("loss")
    train_loss_ax.set_ylim(0, 4)  # 设置部分坐标轴范围
    test_loss_ax.set_ylim(0, 4)  # 设置部分坐标轴范围

    # 均匀分布初始化
    label = "uniform init"
    print(label)
    train_loader, test_x, test_y = generate_dataset()
    model = Net_uniform_init()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    train_loss_list, test_loss_list, precision_list, iteration_num_list = train(
        model, optimizer, criterion, train_loader, test_x, test_y
    )
    pre_ax.plot(iteration_num_list, precision_list, label=label, linewidth=0.5)  # 绘制
    train_loss_ax.plot(iteration_num_list, train_loss_list, label=label, linewidth=0.5)
    test_loss_ax.plot(iteration_num_list, test_loss_list, label=label, linewidth=0.5)

    # 正态分布初始化
    label = "normal init"
    print(label)
    train_loader, test_x, test_y = generate_dataset()
    model = Net_normal_init()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    train_loss_list, test_loss_list, precision_list, iteration_num_list = train(
        model, optimizer, criterion, train_loader, test_x, test_y
    )
    pre_ax.plot(iteration_num_list, precision_list, label=label, linewidth=0.5)
    train_loss_ax.plot(iteration_num_list, train_loss_list, label=label, linewidth=0.5)
    test_loss_ax.plot(iteration_num_list, test_loss_list, label=label, linewidth=0.5)

    # Xavier分布初始化
    label = "xavier init"
    print(label)
    train_loader, test_x, test_y = generate_dataset()
    model = Net_Xavier_init()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    train_loss_list, test_loss_list, precision_list, iteration_num_list = train(
        model, optimizer, criterion, train_loader, test_x, test_y
    )
    pre_ax.plot(iteration_num_list, precision_list, label=label, linewidth=0.5)
    train_loss_ax.plot(iteration_num_list, train_loss_list, label=label, linewidth=0.5)
    test_loss_ax.plot(iteration_num_list, test_loss_list, label=label, linewidth=0.5)

    pre_ax.legend(fontsize=8)
    train_loss_ax.legend(fontsize=8)
    test_loss_ax.legend(fontsize=8)
    plt.tight_layout()
    plt.show()
