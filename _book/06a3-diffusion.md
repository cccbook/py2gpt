# Diffusion Model 影像生成

* [Midjourney V5來了！透過 AI 繪圖原理，你就知道怎麼訓練做圖生成！](https://www.youtube.com/watch?v=EpoE7D2wOeU)

* [What are Diffusion Models?](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)

* [由浅入深了解Diffusion Model](https://zhuanlan.zhihu.com/p/525106459)

* [李宏毅:淺談圖像生成模型 Diffusion Model 原理](https://www.youtube.com/watch?v=azBugJzmz-o)

## ccc: 說明 diffusion model

ChatGPT:

擴散模型 (Diffusion Model) 是一種基於擴散過程 (Diffusion Process) 的機器學習模型，可以用來建模各種時空領域中的自然現象，如分子運動、熱傳導、流體力學等。其主要思想是利用隨機微分方程 (Stochastic Differential Equation) 將時間連續的物理過程離散化，進而使用機器學習模型對其進行建模和預測。

擴散模型的核心是 Langevin 擴散過程，它是一個包含了擾動項的隨機微分方程，用來描述粒子在一個外力場中的運動行為。擴散模型的目標就是在 Langevin 擴散過程的基礎上學習一個映射，將原始數據映射到一個低維表示空間，同時保持數據的主要特徵。因此，擴散模型通常被稱為自編碼器 (Autoencoder) 的變種。

擴散模型通常使用 PyTorch 或 TensorFlow 等深度學習框架來實現，其中 Langevin 擴散過程是以梯度下降的方式進行優化的。模型的訓練過程主要包括三個步驟：預處理數據、設計模型和訓練模型。在訓練完成後，模型可以用來生成新的樣本數據、進行數據降維和特徵提取等操作。

## ccc: diffusion model 的數學公式，用 latex 呈現，但不要加 $ 去 rendering

ChatGPT:

以下是擴散模型的 Langevin 擴散過程的數學公式，其中 $x(t)$ 是時間 $t$ 的樣本數據，$z(t)$ 是擴散模型的低維表示：

$$
\begin{aligned}
\frac{dx(t)}{dt} &= f(x(t)) + \sqrt{2D} \cdot \eta(t) \
\frac{dz(t)}{dt} &= g(z(t)) + \sqrt{2D} \cdot \epsilon(t)
\end{aligned}
$$

其中 $f(\cdot)$ 和 $g(\cdot)$ 是非線性函數，$D$ 是擾動強度，$\eta(t)$ 和 $\epsilon(t)$ 是標準白噪聲。擴散模型通過設計適當的 $f(\cdot)$ 和 $g(\cdot)$，以及適當的擾動強度 $D$，可以實現不同的數據建模和生成任務。


## ccc: 請給一個 Pytorch 的 diffusion model 程式範例

ChatGPT

以下是一個使用 PyTorch 實現的簡單擴散模型的示例，其中採用了標準的 Langevin 擴散過程。這裡使用了隨機梯度下降來進行梯度優化。

```py
import torch
import torch.nn as nn
import torch.optim as optim


class DiffusionModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, time_steps, dt):
        super().__init__()
        
        # 模型參數
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.time_steps = time_steps
        self.dt = dt
        
        # 模型網絡層
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Langevin 擴散過程
        for t in range(self.time_steps):
            dx = -self.fc2(torch.tanh(self.fc1(x))) / self.dt + torch.randn_like(x) * torch.sqrt(2 * self.dt)
            x = x + dx
        
        return x


# 超參數設置
input_size = 1  # 輸入數據的維度
hidden_size = 128  # 隱藏層維度
output_size = 1  # 輸出數據的維度
time_steps = 100  # Langevin 擴散的步數
dt = 1e-2  # Langevin 擴散的時間間隔
batch_size = 128  # 每次訓練使用的批次大小
learning_rate = 1e-3  # 學習率
num_epochs = 10  # 訓練輪數


# 載入數據集
x = torch.randn(10000, input_size)
y = x + 0.1 * torch.randn_like(x)


# 初始化模型和優化器
model = DiffusionModel(input_size, hidden_size, output_size, time_steps, dt)
optimizer = optim.SGD(model.parameters(), lr=learning_rate)


# 訓練模型
for epoch in range(num_epochs):
    for i in range(0, len(x), batch_size):
        # 取出一批數據
        x_batch = x[i:i+batch_size]
        y_batch = y[i:i+batch_size]
        
        # 預測輸出
        y_pred = model(x_batch)
        
        # 計算損失
        loss = torch.mean(torch.pow(y_batch - y_pred, 2))
        
        # 反向傳播並優化參數
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
    # 計算當前的損失值
    with torch.no_grad():
        y_pred = model(x)
        loss = torch.mean(torch.pow(y - y_pred, 2))
        
    print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num

```