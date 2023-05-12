
* [Autoencoders - EXPLAINED](https://www.youtube.com/watch?v=7mRfwaGGAPg)

* [AutoEncoder (一)-認識與理解](https://medium.com/ml-note/autoencoder-%E4%B8%80-%E8%AA%8D%E8%AD%98%E8%88%87%E7%90%86%E8%A7%A3-725854ab25e8)
    * ccc: 除了用 dimension reduction 的想法，應該還可以用 clustering 的想法去想，類似 KMean ....
* [AutoEncoder 之術式解析](https://ithelp.ithome.com.tw/articles/10206869)
* [Keras: 自动编码器：各种各样的自动编码器](https://keras-cn.readthedocs.io/en/latest/legacy/blog/autoencoder/)

## Variational Auto-Encoder

* [Variational Autoencoders - EXPLAINED!](https://www.youtube.com/watch?v=fcvYpzHmhvA)

* [变分自动编码器(VAE)](https://www.fanyeong.com/2018/06/01/vae/)
* [DL、ML筆記(12):Variational AutoEncoder (VAE)](https://jianjiesun.medium.com/dl-ml%E7%AD%86%E8%A8%98-12-variational-autoencoder-vae-6d74bf83daa)
* [Wikipedia:變分自編碼器](https://zh.wikipedia.org/zh-tw/%E5%8F%98%E5%88%86%E8%87%AA%E7%BC%96%E7%A0%81%E5%99%A8)

ccc: Auto Encoder 應該啟發了 Diffusion Model 吧?

## Vector-Quantized Variational AutoEncoder (VQ-VAE)

* [帶你認識Vector-Quantized Variational AutoEncoder - 理論篇](https://medium.com/ai-academy-taiwan/%E5%B8%B6%E4%BD%A0%E8%AA%8D%E8%AD%98vector-quantized-variational-autoencoder-%E7%90%86%E8%AB%96%E7%AF%87-49a1829497bb)
* [帶你認識Vector-Quantized Variational AutoEncoder — Pytorch實作篇](https://medium.com/ai-academy-taiwan/%E5%B8%B6%E4%BD%A0%E8%AA%8D%E8%AD%98vector-quantized-variational-autoencoder-pytorch%E5%AF%A6%E4%BD%9C%E7%AF%87-f04e302ea67)

* [Transformer Neural Net makes music! (JukeboxAI)](https://www.youtube.com/watch?v=7IEEKvcudrA)

## Hierarchical VQ-VAE

* [强大的NVAE：以后再也不能说VAE生成的图像模糊了](https://zhuanlan.zhihu.com/p/159220222)

* [超越BigGAN，DeepMind提出「史上最强非GAN生成器」VQ-VAE-2](https://www.linkresearcher.com/theses/729e4ed8-044f-449d-a2da-746ca2d98903)

## ccc:解釋 Auto Encoder 

ChatGPT:

自編碼器（Auto Encoder）是一種無監督學習的神經網路模型，用於將輸入數據編碼為一個低維度的向量，並從該向量中重建原始輸入。它通常由兩個部分組成：編碼器和解碼器。

編碼器將輸入數據轉換為低維度的向量，該向量通常比輸入數據的維度低得多。解碼器則將該向量解碼回輸入數據，使其與原始數據盡可能相似。訓練自編碼器的過程是通過最小化重建誤差來學習將輸入數據編碼成為低維度向量和解碼回原始數據的能力。

自編碼器常被用於降維，將高維度數據壓縮為低維度數據，以便於可視化和處理。它也可以用於去噪聲、特徵提取、圖像生成等應用中。近年來，許多變種的自編碼器也被提出，如變分自編碼器（Variational Auto Encoder）和生成對抗自編碼器（Generative Adversarial Auto Encoder）。

## ccc: Auto Encoder 的數學公式，用 latex 原始碼呈現 ，不要加 $ 去 rendering

ChatGPT:

自編碼器的數學公式可以表示為：

編碼器：

$$
z = f(x)
$$

解碼器：

$$
\hat{x} = g(z)
$$

其中，x 是輸入數據，z 是編碼器產生的低維度向量， $\hat{x}$ 是解碼器重建的輸入數據。f 和 g 是神經網絡模型，分別用於編碼和解碼。

自編碼器的訓練過程是通過最小化重建誤差（reconstruction error）來完成的，可以表示為：

$$
L(x, \hat{x}) = \sum_{i=1}^{n}(x_i - \hat{x}_i)^2
$$

其中，n 是輸入數據的維度，$x_i$ 和 $\hat{x}_i$ 分別表示輸入數據和重建數據的第 i 個元素。通過最小化重建誤差，自編碼器可以學習到如何將輸入數據編碼為一個低維度向量，同時保留輸入數據的重要特徵。


## Tensorflow + Keras

```py
import tensorflow as tf

# 定義編碼器和解碼器的神經網絡模型
encoder = tf.keras.models.Sequential([
    tf.keras.layers.Dense(256, activation='relu', input_shape=(784,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
])
decoder = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(32,)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(784, activation='sigmoid'),
])

# 定義自編碼器模型
autoencoder = tf.keras.models.Sequential([
    encoder,
    decoder,
])

# 定義訓練過程
autoencoder.compile(optimizer='adam', loss='mse')

# 載入 MNIST 數據集
(x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
x_train = x_train.reshape((len(x_train), 784)).astype('float32') / 255.0
x_test = x_test.reshape((len(x_test), 784)).astype('float32') / 255.0

# 訓練自編碼器
autoencoder.fit(x_train, x_train, epochs=10, batch_size=256, shuffle=True, validation_data=(x_test, x_test))

# 使用自編碼器對測試數據進行重建
decoded_imgs = autoencoder.predict(x_test)

```

## pytorch

```py
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
from torchvision import transforms


# 定義編碼器和解碼器
class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, latent_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, latent_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


class Decoder(nn.Module):
    def __init__(self, latent_size, hidden_size, output_size):
        super().__init__()
        self.fc1 = nn.Linear(latent_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x


# 超參數設置
input_size = 784  # 輸入數據的維度
hidden_size = 256  # 編碼器和解碼器的隱藏層維度
latent_size = 32  # 編碼後的低維度向量的維度
batch_size = 128  # 每次訓練使用的批次大小
learning_rate = 1e-3  # 學習率
num_epochs = 10  # 訓練輪數


# 載入數據集
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)


# 初始化編碼器和解碼器
encoder = Encoder(input_size, hidden_size, latent_size)
decoder = Decoder(latent_size, hidden_size, input_size)


# 定義損失函數和優化器
criterion = nn.MSELoss()
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=learning_rate)


# 訓練自編碼器
for epoch in range(num_epochs):
    for i, (images, _) in enumerate(train_loader):
        # 將數據展開成向量
        images = images.view(-1, input_size)
        
        # 進行編碼和解碼
        codes = encoder(images)
        outputs = decoder(codes)
        
        # 計算重建誤

```
