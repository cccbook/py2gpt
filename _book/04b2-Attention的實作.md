# Attention 注意力機制

## 张浩在路上的文章

* [Attention机制的基本思想与实现原理](https://imzhanghao.com/2021/09/01/attention-mechanism/)


语义编码C是由句子Source的每个单词经过Encoder编码产生的，这意味着不论是生成哪个单词，其实句子Source中任意单词对生成某个目标单词来说影响力都是相同的，这是为何说这个模型没有体现出注意力的缘由。这类似于人类看到眼前的画面，但是眼中却没有注意焦点一样。

...

Attention机制其实就是一系列注意力分配系数，也就是一系列权重参数罢了。

...
可以将Attention机制看做软寻址，序列中每一个元素都由key(地址)和value(元素)数据对存储在存储器里，当有query=key的查询时，需要取出元素的value值(也即query查询的attention值)，与传统的寻址不一样，它不是按照地址取出值的，它是通过计算key与query的相似度来完成寻址。这就是所谓的软寻址，它可能会把所有地址(key)的值(value)取出来，上步计算出的相似度决定了取出来值的重要程度，然后按重要程度合并value值得到attention值，此处的合并指的是加权求和。



* [详解Self-Attention和Multi-Head Attention](https://imzhanghao.com/2021/09/15/self-attention-multi-head-attention)

* [Encoder-Decoder简介](https://imzhanghao.com/2021/08/26/encoder-decoder/)

* [Attention Is All You Need -- Transformer](https://imzhanghao.com/2021/09/18/transformer/)


## ccc: 請用 pytorch 實作一個注意力機制的 Attention 層

* https://github.com/pbloem/former/blob/master/former/transformers.py
* https://github.com/pbloem/former/blob/master/former/modules.py

ChatGPT:

以下是使用 PyTorch 實現注意力機制的 Attention 層的示例程式碼：

```py
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn_weights = nn.Linear(hidden_dim, 1, bias=False)
        
    def forward(self, encoder_outputs, hidden_state):
        # encoder_outputs: [batch_size, seq_len, hidden_dim]
        # hidden_state: [batch_size, hidden_dim]
        
        attn_scores = self.attn_weights(encoder_outputs)
        attn_scores = torch.tanh(attn_scores)
        attn_scores = torch.matmul(attn_scores, hidden_state.unsqueeze(2))
        attn_scores = attn_scores.squeeze(2)
        attn_weights = torch.softmax(attn_scores, dim=1)
        context = torch.sum(encoder_outputs * attn_weights.unsqueeze(2), dim=1)
        # context: [batch_size, hidden_dim]
        
        return context, attn_weights

```

在此實現中，我們使用一個線性層來計算注意力權重，並將其應用於輸入序列中的每個時間步。注意力權重由一個雙曲正切激活函數和矩陣乘法計算得到，並通過 softmax 函數歸一化，以產生注意力分配。最後，我們使用注意力權重加權編碼器輸出以生成上下文向量，作為注意力層的輸出。

注意力機制是深度學習中常用的技術，尤其在自然語言處理任務中。它允許模型選擇性地關注輸入中的特定部分，並在某些任務中取得了非常好的表現。

## ccc: 注意力機制的 K, Q, V 在哪裡呢?

注意力機制中的 K、Q、V 是用來計算注意力權重的關鍵元素，它們通常是透過矩陣運算從模型的輸入和/或隱藏狀態中計算得到。

在 Transformer 等模型中，K、Q、V 是從模型的輸入序列和上一層的輸出中計算得到的。在這些模型中，K、Q、V 的計算方式如下：

1. K (Key)：用來計算注意力權重的鍵值，通常是通過線性映射得到的，其維度為 (seq_len, hidden_dim)。

2. Q (Query)：用來計算注意力權重的查詢向量，通常也是通過線性映射得到的，其維度為 (batch_size, hidden_dim)。

3. V (Value)：用來加權編碼器輸出以生成上下文向量的向量，也是通過線性映射得到的，其維度和 K 相同。

在自定義的注意力層中，K、Q、V 的來源可能會有所不同，但是它們的基本計算方式仍然是相似的。例如，在先前提到的注意力層中，Q 就是隱藏狀態，K 和 V 則是來自編碼器輸出。

## ccc: 如何用 pytorch 實現注意力機制的 K, Q, V 呢?

以下是使用 PyTorch 實現注意力機制的 K、Q、V 的示例程式碼：

```py
import torch
import torch.nn as nn

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn_weights = nn.Linear(hidden_dim, 1, bias=False)
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
    def forward(self, encoder_outputs, hidden_state):
        # encoder_outputs: [batch_size, seq_len, hidden_dim]
        # hidden_state: [batch_size, hidden_dim]
        
        Q = self.W_q(hidden_state).unsqueeze(1)
        K = self.W_k(encoder_outputs)
        V = self.W_v(encoder_outputs)
        
        attn_scores = torch.matmul(Q, K.transpose(1, 2))
        attn_scores = torch.softmax(attn_scores, dim=-1)
        context = torch.matmul(attn_scores, V)
        context = context.squeeze(1)
        # context: [batch_size, hidden_dim]
        
        return context, attn_scores

```

在此實現中，我們使用三個線性層來計算注意力機制的 K、Q、V。具體來說，線性層 self.W_q、self.W_k、self.W_v 用於從模型的輸入中計算出查詢向量 Q、鍵值矩陣 K 和值矩陣 V。

然後，我們透過矩陣乘法計算出注意力分數。在這個實現中，我們首先將查詢向量 Q 和鍵值矩陣 K 做矩陣乘法，並通過 softmax 函數歸一化，以產生注意力分配。最後，我們用注意力權重加權值矩陣 V，以生成上下文向量，作為注意力層的輸出。

這個實現中，K、Q、V 的計算方式是比較標準的，可應用於 Transformer 等深度學習模型中。

## ccc: 那如何建構 multi head attention 呢?

ChatGPT:

Multi-head Attention 是一種常用的注意力機制，它能夠提高模型對輸入序列的建模能力，以及對不同特徵子空間的建模能力。下面是使用 PyTorch 實現 Multi-head Attention 的範例程式碼：

```py
import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        
        self.W_q = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_k = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_v = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.W_o = nn.Linear(hidden_dim, hidden_dim, bias=False)
        
    def forward(self, encoder_outputs, hidden_state):
        # encoder_outputs: [batch_size, seq_len, hidden_dim]
        # hidden_state: [batch_size, hidden_dim]
        
        batch_size = encoder_outputs.size(0)
        
        Q = self.W_q(hidden_state).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2) 
        K = self.W_k(encoder_outputs).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2) 
        V = self.W_v(encoder_outputs).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2) 
        
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        attn_scores = torch.softmax(attn_scores, dim=-1)
        
        context = torch.matmul(attn_scores, V)
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.hidden_dim)
        
        output = self.W_o(context)
        
        return output, attn_scores

```

在此實現中，我們首先使用三個線性層來計算注意力機制的 K、Q、V。然後，我們使用 view() 函數將矩陣重塑為形狀 (batch_size, num_heads, seq_len, head_dim)，然後將第二和第三個維度進行轉置。這樣做是為了讓每個頭獨立地學習對輸入序列進行注意力機制。

接下來，我們使用矩陣乘法計算注意力分數，並透過 softmax 函數歸一化。然後，我們用注意力權重加權值矩陣 V，以生成上下文向量。最後，我們將上下文向量重塑為 (batch_size, seq_len, hidden_dim) 的形狀，並通過另一個線性層進行轉換。

這個實現中，K、Q、V 的計算方式與基本的注意力機制相同，只是將輸入序列分為了 num_heads 個子空間，並獨立地對每個子空間進行 K、Q、V 的計算。這樣可以提高模型對輸入序列的建模能力，特別是當輸入序列非常長時。

值得注意的是，在 Multi-head Attention 中，注意力機制的輸出是每個頭的注意力權重和對應的值向量的加權和。因此，我們需要將獨立計算的每個頭的結果進行連接(concatenate)。

最後，返回輸出向量和注意力權重。注意力權重可以用於可視化模型對輸入序列的關注。

