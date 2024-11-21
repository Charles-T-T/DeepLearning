# Lab 5：循环神经网络

## 理论题

**题目1：** 推导公式(6.40)和公式(6.41)中的梯度。

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}} = \sum_{t=1}^T \sum_{k=1}^t \delta_{t, k} \boldsymbol{x}_k^\top, \tag{6.40}
$$

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{b}} = \sum_{t=1}^T \sum_{k=1}^t \delta_{t, k}. \tag{6.41}
$$

**解答：** 

因为

$$
\boldsymbol{z}_k = \boldsymbol{Uh}_{k-1} + \boldsymbol{Wx}_k + \boldsymbol{b}
$$

所以
$$
\begin{align}
\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}} &= \sum_{t=1}^{T} \frac{\partial\mathcal L_t}{\partial\boldsymbol{z}_k} \frac{\partial\boldsymbol{z}_k}{\partial\boldsymbol W} \\
&= \sum_{k=1}^{t} \frac{\partial\mathcal L_t}{\partial\boldsymbol{z}_k} \boldsymbol{x}_k^\top \\
\end{align}
$$

又因为定义了误差项 $\delta_{t,k}$ 为第 $t$ 时刻的损失对第 $k$ 步隐藏神经元的净输入 $\boldsymbol{z}_k$ 的导数，所以上式等于

$$
\sum_{t=1}^T \sum_{k=1}^t \delta_{t, k} \boldsymbol{x}_k^\top, \tag{6.40}
$$

即公式(6.40)

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{W}} = \sum_{t=1}^T \sum_{k=1}^t \delta_{t, k} \boldsymbol{x}_k^\top
$$

同理可得公式(6.41)

$$
\frac{\partial \mathcal{L}}{\partial \boldsymbol{b}} = \sum_{t=1}^T \sum_{k=1}^t \delta_{t, k}
$$



**题目2：** 试验证公式(6.31)中的 $\boldsymbol{z}_k(\boldsymbol{z}_k = \boldsymbol{Uh}_{k-1} + \boldsymbol{Wx}_k + \boldsymbol{b})$ 对 $u_{ij}$ 直接求偏导数 $\frac{\partial^{+}\boldsymbol{z}_k}{\partial u_{ij}}$ 等价于递归下去对 $h_{k-1}$ 接着求导。
$$
\begin{equation}
\frac{\partial \mathcal{L}_t}{\partial u_{ij}} = \sum_{k=1}^t \frac{\partial^+ \boldsymbol{z}_k}{\partial u_{ij}} \frac{\partial \mathcal{L}_t}{\partial \boldsymbol{z}_k} \tag{6.31}
\end{equation}
$$

**解答：**

由于隐状态 $\boldsymbol h_{k-1}$ 依赖于前一步的隐状态 $\boldsymbol h_{k-2}$ ，故可以通过链式法则递归求导。由于

$$
\boldsymbol{z}_k = \boldsymbol{Uh}_{k-1} + \boldsymbol{Wx}_k + \boldsymbol{b}, \\
\boldsymbol{h}_{k-1} = activation(\boldsymbol z_{k-1})
$$

当递归传播到第 $k-1$ 步的时候：

- 对 $\boldsymbol z_k$ 求导需要用到隐状态梯度： $\frac{\partial\boldsymbol z_k}{\partial\boldsymbol h_{k-1}} = \boldsymbol{U}$ ，表明当前时间步的梯度将被传递到 $\boldsymbol{U}$ 上
- 递归对 $\boldsymbol{h}_{k-2}$ 求导时，将通过上一步 $\boldsymbol{z}_{k-2}$ 的依赖项继续传播
- 最终，完整的梯度展开形式为

$$
\frac{\partial \mathcal{L}_t}{\partial u_{ij}} = \sum_{k=1}^t \frac{\partial\mathcal{L}_t}{\partial\boldsymbol{z}_k} \frac{\partial\boldsymbol{z}_k}{\partial\boldsymbol{h}_{k-1}} \dots \frac{\partial\boldsymbol{z}_1}{\partial{u}_{ij}}
$$

其中每一步递归传递都是由隐状态的依赖性实现的。故总的来说，递归求导过程中， $\boldsymbol h_{k-1}$ 的梯度被依赖结构传播到 $\boldsymbol z_k$ ，最终结果在数值上等价于直接求导的结果。 

## 代码题

### 问题描述
利用循环神经网络(LSTM)，实现简单的古诗生成任务

### 代码补全

