# ccc: 深度學習中的 Embed 層是甚麼？

其實問 ChatGPT 問不出結果，看這篇比較快

* https://jaketae.github.io/study/word2vec/


forward:

```py

def init_network(vocab_size, n_embedding):
    model = {
        "w1": np.random.randn(vocab_size, n_embedding),
        "w2": np.random.randn(n_embedding, vocab_size)
    }
    return model

def forward(model, X, return_cache=True):
    cache = {}
    
    cache["a1"] = X @ model["w1"]
    cache["a2"] = cache["a1"] @ model["w2"]
    cache["z"] = softmax(cache["a2"])
    
    if not return_cache:
        return cache["z"]
    return cache
```

forward:

$A_1 = X W_1$

$A_2 = A_1 W_2$

$Z = softmax(A_2)$

所以 Embedding model 是一個將vocab_size 大小的輸入， 用 W1 編碼成 n_embedding 大小的向量。

然後用 W2 將其還原，並讓最後輸出的 Z 盡可能就是原來的 one-hot encoding 的那種網路。

其 loss 的定義如下：

$\frac{\partial L}{\partial A_2} = Z-y$


所以反傳遞的程式碼如下

```py
def backward(model, X, y, alpha):
    cache  = forward(model, X)
    da2 = cache["z"] - y
    dw2 = cache["a1"].T @ da2
    da1 = da2 @ model["w2"].T
    dw1 = X.T @ da1
    assert(dw2.shape == model["w2"].shape)
    assert(dw1.shape == model["w1"].shape)
    model["w1"] -= alpha * dw1
    model["w2"] -= alpha * dw2
    return cross_entropy(cache["z"], y)
```

問題是，還原之後不是應該直接變回 X 嗎？為何 loss 是用 z 和 y 來做 cross_entropy， y 是甚麼？

那個 y 是詞彙 x 的上下文，也就是前後的詞彙，如下圖所示。

![](https://miro.medium.com/max/1400/1*Mmp1vbFOxrmiCF17lYJWRA.png)

所以還原後，要讓 z 在上下文關係當中，預測能力和 x 愈像愈好


# ccc: 深度學習中的 Embed 層是甚麼？

ChatGPT:

Embed層是深度學習中一種常用的層，它主要用於將高維度的稀疏資料（如文字、ID等）轉換成低維度的密集向量，通常稱為嵌入向量（Embedding Vector）。

在自然語言處理（NLP）等應用中，我們需要將文字轉換成向量，以便於進行數據處理和分析。一個常見的方法是使用 one-hot encoding，即將每個文字表示成一個高維度的向量，其中大部分元素都是0，只有一個元素是1，代表該文字的位置。然而，這樣表示的向量維度很高，會導致模型較難訓練。因此，Embed層的作用就是將高維度的稀疏資料轉換成低維度的密集向量，可以減少向量維度，並且在保留重要信息的同時，使相似的文字在向量空間中距離更接近。

Embed層通常是在模型的輸入層中使用的，它會接收一個高維度的向量，並返回一個低維度的嵌入向量。Embed層的參數是可訓練的，模型會通過反向傳播算法來學習如何將高維度的稀疏資料轉換成低維度的嵌入向量。
