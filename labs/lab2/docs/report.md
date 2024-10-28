# Lab2 前馈神经网络

:man_student: Charles

1. **试设计一个前馈神经网络来解决 XOR 问题，要求该前馈神经网络具有两个隐藏神经元和一个输出神经元，并使用 `ReLU` 作为激活函数。**

   ```python
   import torch
   import torch.nn as nn
   import torch.optim as optim
   
   # 数据准备（xor计算的四种输入输出）
   x = torch.tensor([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=torch.float32)
   y = torch.tensor([[0], [1], [0], [1]], dtype=torch.float32)
   
   # 定义前馈神经网络模型
   class XORModel(nn.Module):
       def __init__(self):
           super(XORModel, self).__init__()
           self.fc1 = nn.Linear(2, 2)  # 两个隐藏神经元
           self.fc2 = nn.Linear(2, 1)  # 一个输出神经元
           self.relu = nn.ReLU()
           self.sigmoid = nn.Sigmoid()
   
       def forward(self, x):
           x = self.relu(self.fc1(x))  # 隐藏层用ReLU作为激活函数
           x = self.sigmoid(self.fc2(x))  # 输出层用Sigmoid
           return x
   
   model = XORModel()
   criterion = nn.BCELoss()  # 损失函数
   optimizer = optim.SGD(model.parameters(), lr=0.01)  # 优化器
   
   # 训练模型
   epochs = 10000
   for epoch in range(epochs):
       model.train()
       optimizer.zero_grad()
       y_pred = model(x)
       loss = criterion(y_pred, y)
       loss.backward()
       optimizer.step()
       
       if epoch == 0 or (epoch + 1) % 1000 == 0:
           print(f'Epoch[{epoch + 1}], Loss: {loss.item():.4f}')
       
   # 测试模型
   model.eval()
   with torch.no_grad():
       y_pred = torch.round(model(x))  # 四舍五入，将概率映射为结果0或1
       print(f'Prediction:\n {y_pred}')
       print(f'Actual:\n {y}')
   ```

   输出：

   <img src="./output" alt="output" style="zoom: 67%;" /> 

   可以看到，训练后的前馈神经网络模型成功解决了XOR问题。

   

2. **如果限制一个神经网络的总神经元数量（不考虑输入层）为 $N+1$ ，输入层大小为 $M_0$ ，输出层大小为 $1$  ，隐藏层的层数为 $L$ ，每个隐藏层的神经元数量为 $N/L$ ，试分析参数数量和隐藏层层数 $L$ 的关系。**

- 从输入层到第一个隐藏层， $M_0$ 个神经元和 $N/L$ 个神经元全连接，对应参数数量为：

$$
M_0 \times N/L
$$
     
- 再加上第一个隐藏层的 $N/L$ 个偏置参数，则这部分的总参数数量为：
     
$$
(M_0 + 1) \times N/L
$$
     
   - 从第一个隐藏层到最后一个隐藏层，共经历 $L - 1$ 次隐藏层神经元之间的全连接，每次对应参数数量为 $(N/L)^2$ ，加上偏置参数 $N/L$ 个，则这部分的总参数数量为：

$$
(L - 1) \times ((N/L)^2 + N/L) = (L-1) \times N/L \times (N/L - 1)
$$
     
   - 从最后一个隐藏层到输出层， $N/L$ 个隐藏层神经元与 $1$ 个输出神经元连接，加上输出神经元的 $1$ 个偏置参数，则这部分的总参数数量为：

$$
N/L + 1
$$
     

   综上所述，该神经网络中参数数量 $P$ 与隐藏层层数 $L$ 的关系为：

$$
   \begin{align}
   P(L) &= (M_0 + 1) \times N/L + (L+1) \times N/L \times (N/L + 1) + N/L + 1 \\
   &= \frac{N}{L} [(M_0 + 2) + (L + 1)(\frac{N}{L} + 1)] + 1
   \end{align}
$$
   

3. **为什么在神经网络模型的结构化风险函数中不对偏置 $b$ 进行正则化？**

   神经网络模型采用结构化风险函数时，只对权重 $W$ 进行正则化，目的是通过惩罚 $W$ 来降低模型的复杂度，避免训练时模型对某些特征依赖过高和过拟合。而偏置 $b$ 不会影响模型对各特征的依赖，只是对模型输出作整体的平移，故对它进行正则化并不能起到降低模型复杂度或者防止过拟合等作用，也就没必要正则化了。

   

4. **梯度消失问题是否可以通过增加学习率来缓解？**

   不可以。梯度消失的主要原因在于神经网络层数过深或某些激活函数梯度极小，即神经网络模型自身的结构问题。单纯增加学习率虽然可以提高参数更新的速度，但是对于趋于0（“消失”）的梯度，参数的变化仍可能非常微小。而且过高的学习率可能导致梯度爆炸，使模型训练不稳定、发散等，不利于训练效果。
