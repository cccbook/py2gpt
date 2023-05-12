# Pytorch 中的 Transformer

* [pytorch中的transformer](https://zhuanlan.zhihu.com/p/107586681)

* https://github.com/pytorch/examples/blob/main/word_language_model/main.py (重要)
    * 之後要仔細閱讀

## nn.Transformer

torch.nn.Transformer(d_model=512, nhead=8, num_encoder_layers=6, 
num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, 
activation='relu', custom_encoder=None, custom_decoder=None)

参数：

```
d_model –编码器/解码器输入大小（默认 512）。
nhead –多头注意力模型的头数（默认为8）。
num_encoder_layers –编码器中子编码器层的数量（默认为6）。
num_decoder_layers –解码器中子解码器层的数量（默认为6）。
dim_feedforward –前馈网络模型的中间层维度（默认= 2048）。
dropout –默认值= 0.1。
activation–编码器/解码器中间层的激活函数，relu或gelu（默认值= relu）。
custom_encoder –自定义编码器（默认=None）。
custom_decoder –自定义解码器（默认=None）。
```

例子：

```
>>> transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
>>> src = torch.rand((10, 32, 512)) # 輸入長度 10，Batch Size 32，embed 詞向量大小 512
>>> tgt = torch.rand((20, 32, 512)) # 輸出長度 20，Batch Size 32，embed 詞向量大小 512
>>> out = transformer_model(src, tgt)
```



forward(src, tgt, src_mask=None, tgt_mask=None, memory_mask=None, 
src_key_padding_mask=None, tgt_key_padding_mask=None, memory_key_padding_mask=None)

```
src – the sequence to the encoder (required).
tgt – the sequence to the decoder (required).
src_mask – the additive mask for the src sequence (optional).
tgt_mask – the additive mask for the tgt sequence (optional).
memory_mask – the additive mask for the encoder output (optional).
src_key_padding_mask – the ByteTensor mask for src keys per batch (optional).
tgt_key_padding_mask – the ByteTensor mask for tgt keys per batch (optional).
memory_key_padding_mask – the ByteTensor mask for memory keys per batch (optional).
```


Shape:

src: (S, N, E) : S是 src 的序列长度，N batch size，E 特征维度
tgt: (T, N, E) : T是 tgt 的序列長度，N batch size, E 特征维度
src_mask: (S, S)(S,S ) .
tgt_mask: (T, T)(T,T ) .
memory_mask: (T, S)(T,S ) .
src_key_padding_mask: (N, S)(N,S ) .
tgt_key_padding_mask: (N, T)(N,T ) .
memory_key_padding_mask: (N, S)(N,S ) .
