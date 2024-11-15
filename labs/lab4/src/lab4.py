# 待补全代码

import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import torch.nn.functional as F
import numpy as np


# 参数
learning_rate = 1e-4
dropout_rate = 0.5
max_epoch = 3
BATCH_SIZE = 50

my_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


'''
1.调用torchvision.datasets.MNIST 读取MNIST数据集 将数据包装为Dataset类
2.通过DataLoader将dataset变量变为迭代器
3.同样的方法处理训练和测试数据，设置BACTH_SIZE，思考train和test的时候是否需要shuffle
'''

# 下载MNIST数据集
DOWNLOAD_MNIST = False
if not (os.path.exists('./mnist/')) or not os.listdir('./mnist/'):
    DOWNLOAD_MNIST = True

train_data = torchvision.datasets.MNIST(
    root='./mnist/', train=True, transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST)
train_loader = Data.DataLoader(
    dataset=train_data, batch_size=BATCH_SIZE, shuffle=True)

test_data = torchvision.datasets.MNIST(
    root='./mnist/', train=False, transform=torchvision.transforms.ToTensor(), download=DOWNLOAD_MNIST)
test_loader = Data.DataLoader(
    dataset=test_data, batch_size=BATCH_SIZE, shuffle=False)


'''
写一个卷积网络
    __init__: 初始化模型的地方，在这里声明模型的结构
    forward: 调用模型进行计算，输入为x（按照batch_size组织好的样本数据），输出为模型预测结果
'''


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=32, kernel_size=7, padding=3, device=my_device)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=5, padding=0, device=my_device)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc_3 = nn.Linear(64 * 5 * 5, 1024, device=my_device)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fc_4 = nn.Linear(1024, 10, device=my_device)

    def forward(self, x):
        out = self.conv1(x)
        out = self.pool1(out)
        out = self.conv2(out)
        out = self.pool2(out)
        out = out.view(-1, 64 * 5 * 5)  # flatten
        out = self.fc_3(out)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.fc_4(out)

        # out为x经过一系列计算后最后一层fc_4输出的logit
        output = F.softmax(out, dim=1)
        return output


# 测试 输入模型，遍历test_loader，输出模型的准确率
# 注意 在测试时需要禁止梯度回传 可以考虑eval()模式和torch.no_grad()。他们有区别吗？
def test(cnn):
    cnn.eval()
    test_correct = 0

    with torch.no_grad():
        for (images, labels) in test_loader:
            images, labels = images.to(my_device), labels.to(my_device)
            probs = cnn(images)
            _, predicted_labels = torch.max(probs, dim=1)
            test_correct += (predicted_labels == labels).sum().item()

    # 计算模型正确率
    return float(100 * test_correct / len(test_data))


# 训练
def train(cnn):
    # 声明一个Adam的优化器
    optimizer = torch.optim.Adam(params=cnn.parameters(), lr=learning_rate)
    loss_func = nn.CrossEntropyLoss()  # 使用CrossEntropyLoss

    cnn.train()  # 将模型调节为训练模式
    for epoch in range(max_epoch):
        total_loss = 0
        for step, (x, y) in enumerate(train_loader):
            x, y = x.to(my_device), y.to(my_device)
            optimizer.zero_grad()  # 清空梯度
            probs = cnn(x)  # 得到模型针对训练数据x的概率分布
            loss = loss_func(probs, y)  # 计算loss
            loss.backward()  # 梯度回传
            optimizer.step()  # 更新优化器
            total_loss += loss

            if step != 0 and step % 20 == 0:
                print("=" * 10, step, "=" * 10,
                      "test accuracy is", test(cnn), "%", "=" * 10)
                print(f"loss: {loss:.4f}")

        # avg_train_loss = total_loss / len(train_loader)
        # accuracy = test(cnn)
        # print(f"Epoch {epoch}, Train Loss: {avg_train_loss:.4f}, Test Acc: {accuracy}%")


if __name__ == '__main__':
    cnn = CNN()
    train(cnn)
