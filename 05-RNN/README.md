# RNN -- get embed

取得詞彙對應的 embed ，並計算兩兩之間的相似度

```
$ python main.py english gru train
tokens= 5231
len(ids)= 5231
ids.size(0)= 5231
batch_size= 20
num_batches= 261
len(ids)= 5220
ids.shape= torch.Size([20, 261])
dictionary= {0: 'the', 1: 'little', 2: 'pig', 3: '<eos>', 4: 'every', 5: 'white', 6: 'cat', 7: 'chase', 8: 'a', 9: 'bite', 10: 'black', 11: 'dog', 12: 'love'}
vocab_size= 13
Epoch [1/20], Step[0/8], Loss: 2.5108, Perplexity: 12.31
Epoch [2/20], Step[0/8], Loss: 2.4390, Perplexity: 11.46
Epoch [3/20], Step[0/8], Loss: 2.3731, Perplexity: 10.73
Epoch [4/20], Step[0/8], Loss: 2.3101, Perplexity: 10.08
Epoch [5/20], Step[0/8], Loss: 2.2116, Perplexity:  9.13
Epoch [6/20], Step[0/8], Loss: 2.0648, Perplexity:  7.88
Epoch [7/20], Step[0/8], Loss: 1.8698, Perplexity:  6.49
Epoch [8/20], Step[0/8], Loss: 1.6856, Perplexity:  5.40
Epoch [9/20], Step[0/8], Loss: 1.5457, Perplexity:  4.69
Epoch [10/20], Step[0/8], Loss: 1.4466, Perplexity:  4.25
Epoch [11/20], Step[0/8], Loss: 1.3853, Perplexity:  4.00
Epoch [12/20], Step[0/8], Loss: 1.3503, Perplexity:  3.86
Epoch [13/20], Step[0/8], Loss: 1.3294, Perplexity:  3.78
Epoch [14/20], Step[0/8], Loss: 1.3155, Perplexity:  3.73
Epoch [15/20], Step[0/8], Loss: 1.3065, Perplexity:  3.69
Epoch [16/20], Step[0/8], Loss: 1.3005, Perplexity:  3.67
Epoch [17/20], Step[0/8], Loss: 1.2959, Perplexity:  3.65
Epoch [18/20], Step[0/8], Loss: 1.2923, Perplexity:  3.64
Epoch [19/20], Step[0/8], Loss: 1.2894, Perplexity:  3.63
Epoch [20/20], Step[0/8], Loss: 1.2871, Perplexity:  3.62

ccckmit@asus MINGW64 /d/ccc/py2cs/03-人工智慧/05-深度學習/04-rnn/04-rnn4embed (master)
$ python main.py english gru test
tokens= 5231
len(ids)= 5231
ids.size(0)= 5231
batch_size= 20
num_batches= 261
len(ids)= 5220
ids.shape= torch.Size([20, 261])
dictionary= {0: 'the', 1: 'little', 2: 'pig', 3: '<eos>', 4: 'every', 5: 'white', 6: 'cat', 7: 'chase', 8: 'a', 9: 'bite', 10: 'black', 11: 'dog', 12: 'love'}
vocab_size= 13
Sampled [100/1000] words and save to english_gru.txt
Sampled [200/1000] words and save to english_gru.txt
Sampled [300/1000] words and save to english_gru.txt
Sampled [400/1000] words and save to english_gru.txt
Sampled [500/1000] words and save to english_gru.txt
Sampled [600/1000] words and save to english_gru.txt
Sampled [700/1000] words and save to english_gru.txt
Sampled [800/1000] words and save to english_gru.txt
Sampled [900/1000] words and save to english_gru.txt
Sampled [1000/1000] words and save to english_gru.txt

ccckmit@asus MINGW64 /d/ccc/py2cs/03-人工智慧/05-深度學習/04-rnn/04-rnn4embed (master)
$ python main.py english gru embed
tokens= 5231
len(ids)= 5231
ids.size(0)= 5231
batch_size= 20
num_batches= 261
len(ids)= 5220
ids.shape= torch.Size([20, 261])
dictionary= {0: 'the', 1: 'little', 2: 'pig', 3: '<eos>', 4: 'every', 5: 'white', 6: 'cat', 7: 'chase', 8: 'a', 9: 'bite', 10: 'black', 11: 'dog', 12: 'love'}
vocab_size= 13
0:the tensor([-0.4581,  0.5144,  0.9367,  0.2073,  1.3344, -1.5348, -0.6420,  2.0518],
       grad_fn=<SelectBackward0>)
1:little tensor([ 0.0546,  0.2247, -1.0713,  0.1466, -1.1536, -0.3864,  0.7228,  2.8768],
       grad_fn=<SelectBackward0>)
2:pig tensor([ 2.0651,  1.3524, -0.0919, -1.4458, -0.1625,  1.8365,  1.5188,  0.6456],
       grad_fn=<SelectBackward0>)
3:<eos> tensor([-1.1171,  0.6835, -1.2271, -2.3023, -0.4831, -0.5877,  1.0612, -0.6222],
       grad_fn=<SelectBackward0>)
4:every tensor([-0.4800,  0.3519,  0.3203,  0.0918, -1.0742, -1.1115,  1.6148, -0.2141],
       grad_fn=<SelectBackward0>)
5:white tensor([-0.8440, -0.4340, -1.5399,  1.0683,  0.8548,  0.5559, -1.2158,  1.0957],
       grad_fn=<SelectBackward0>)
6:cat tensor([ 0.5266,  0.8549, -0.4540, -0.9064, -1.0882,  0.4731,  0.0770, -0.2815],
       grad_fn=<SelectBackward0>)
7:chase tensor([ 0.9403,  2.3334,  0.8292,  0.0470,  0.5718,  0.0995, -0.8233, -0.1684],
       grad_fn=<SelectBackward0>)
8:a tensor([ 0.4237, -0.5010, -0.6227, -0.4767,  1.8748,  0.3109,  1.6325, -1.3139],
       grad_fn=<SelectBackward0>)
9:bite tensor([-1.3278,  0.0203, -0.1235, -0.2793,  1.9733, -0.5440, -0.7607, -1.0597],
       grad_fn=<SelectBackward0>)
10:black tensor([-1.0865, -1.2685,  0.4613, -0.7811,  0.4690, -0.2325, -0.5520,  1.0553],
       grad_fn=<SelectBackward0>)
11:dog tensor([-0.0929, -0.0272,  1.1641, -1.6313,  0.1407,  1.1421, -1.6248,  0.4742],
       grad_fn=<SelectBackward0>)
12:love tensor([-1.9701,  0.4280, -0.4474, -0.6108, -1.4886, -0.7862, -1.3717,  0.1924],
       grad_fn=<SelectBackward0>)
0:the
  similarity 0:the 1.0
  similarity 1:little 0.3340418338775635
  similarity 2:pig -0.27555009722709656
  similarity 3:<eos> -0.23734737932682037
  similarity 4:every -0.06464453041553497
  similarity 5:white 0.248567134141922
  similarity 6:cat -0.5379056930541992
  similarity 7:chase 0.25968217849731445
  similarity 8:a -0.2977064251899719
  similarity 9:bite 0.2511136829853058
  similarity 10:black 0.49293383955955505
  similarity 11:dog 0.13488972187042236
  similarity 12:love 0.10986379534006119
1:little
  similarity 0:the 0.3340418338775635
  similarity 1:little 1.0
  similarity 2:pig 0.2132406234741211
  similarity 3:<eos> 0.07522256672382355
  similarity 4:every 0.24414226412773132
  similarity 5:white 0.2828158736228943
  similarity 6:cat 0.1401294320821762
  similarity 7:chase -0.2163507044315338
  similarity 8:a -0.428866446018219
  similarity 9:bite -0.5975844860076904
  similarity 10:black 0.15780359506607056
  similarity 11:dog -0.19662168622016907
  similarity 12:love 0.18874666094779968
2:pig
  similarity 0:the -0.27555009722709656
  similarity 1:little 0.2132406234741211
  similarity 2:pig 1.0
  similarity 3:<eos> 0.18399670720100403
  similarity 4:every -0.025897560641169548
  similarity 5:white -0.36983588337898254
  similarity 6:cat 0.6439206004142761
  similarity 7:chase 0.3429379165172577
  similarity 8:a 0.24965402483940125
  similarity 9:bite -0.5182961821556091
  similarity 10:black -0.40537169575691223
  similarity 11:dog 0.1785326451063156
  similarity 12:love -0.4939063489437103
3:<eos>
  similarity 0:the -0.23734737932682037
  similarity 1:little 0.07522256672382355
  similarity 2:pig 0.18399670720100403
  similarity 3:<eos> 1.0000001192092896
  similarity 4:every 0.41741517186164856
  similarity 5:white -0.2836710214614868
  similarity 6:cat 0.513802707195282
  similarity 7:chase -0.1821485161781311
  similarity 8:a 0.255853533744812
  similarity 9:bite 0.16655325889587402
  similarity 10:black 0.03286418318748474
  similarity 11:dog -0.03706585615873337
  similarity 12:love 0.4061325788497925
4:every
  similarity 0:the -0.06464453041553497
  similarity 1:little 0.24414226412773132
  similarity 2:pig -0.025897560641169548
  similarity 3:<eos> 0.41741517186164856
  similarity 4:every 1.0000001192092896
  similarity 5:white -0.5792595148086548
  similarity 6:cat 0.14676016569137573
  similarity 7:chase -0.20688076317310333
  similarity 8:a -0.00918348878622055
  similarity 9:bite -0.2965325117111206
  similarity 10:black -0.2241065800189972
  similarity 11:dog -0.577937662601471
  similarity 12:love 0.1548893004655838
5:white
  similarity 0:the 0.248567134141922
  similarity 1:little 0.2828158736228943
  similarity 2:pig -0.36983588337898254
  similarity 3:<eos> -0.2836710214614868
  similarity 4:every -0.5792595148086548
  similarity 5:white 0.9999998807907104
  similarity 6:cat -0.40204858779907227
  similarity 7:chase -0.20656490325927734
  similarity 8:a -0.1561886966228485
  similarity 9:bite 0.2710902690887451
  similarity 10:black 0.3078344464302063
  similarity 11:dog -0.023855343461036682
  similarity 12:love 0.1925353854894638
6:cat
  similarity 0:the -0.5379056930541992
  similarity 1:little 0.1401294320821762
  similarity 2:pig 0.6439206004142761
  similarity 3:<eos> 0.513802707195282
  similarity 4:every 0.14676016569137573
  similarity 5:white -0.40204858779907227
  similarity 6:cat 0.9999997615814209
  similarity 7:chase 0.27744749188423157
  similarity 8:a -0.15711557865142822
  similarity 9:bite -0.485257625579834
  similarity 10:black -0.489495187997818
  similarity 11:dog 0.18695366382598877
  similarity 12:love 0.203840434551239
7:chase
  similarity 0:the 0.25968217849731445
  similarity 1:little -0.2163507044315338
  similarity 2:pig 0.3429379165172577
  similarity 3:<eos> -0.1821485161781311
  similarity 4:every -0.20688076317310333
  similarity 5:white -0.20656490325927734
  similarity 6:cat 0.27744749188423157
  similarity 7:chase 1.0
  similarity 8:a -0.15564732253551483
  similarity 9:bite 0.07113300263881683
  similarity 10:black -0.4761693477630615
  similarity 11:dog 0.26908931136131287
  similarity 12:love -0.12480109930038452
8:a
  similarity 0:the -0.2977064251899719
  similarity 1:little -0.428866446018219
  similarity 2:pig 0.24965402483940125
  similarity 3:<eos> 0.255853533744812
  similarity 4:every -0.00918348878622055
  similarity 5:white -0.1561886966228485
  similarity 6:cat -0.15711557865142822
  similarity 7:chase -0.15564732253551483
  similarity 8:a 1.0
  similarity 9:bite 0.3962849974632263
  similarity 10:black -0.17618079483509064
  similarity 11:dog -0.3050384819507599
  similarity 12:love -0.651657223701477
9:bite
  similarity 0:the 0.2511136829853058
  similarity 1:little -0.5975844860076904
  similarity 2:pig -0.5182961821556091
  similarity 3:<eos> 0.16655325889587402
  similarity 4:every -0.2965325117111206
  similarity 5:white 0.2710902690887451
  similarity 6:cat -0.485257625579834
  similarity 7:chase 0.07113300263881683
  similarity 8:a 0.3962849974632263
  similarity 9:bite 1.0000001192092896
  similarity 10:black 0.30127352476119995
  similarity 11:dog 0.1033247634768486
  similarity 12:love 0.13834749162197113
10:black
  similarity 0:the 0.49293383955955505
  similarity 1:little 0.15780359506607056
  similarity 2:pig -0.40537169575691223
  similarity 3:<eos> 0.03286418318748474
  similarity 4:every -0.2241065800189972
  similarity 5:white 0.3078344464302063
  similarity 6:cat -0.489495187997818
  similarity 7:chase -0.4761693477630615
  similarity 8:a -0.17618079483509064
  similarity 9:bite 0.30127352476119995
  similarity 10:black 1.0000001192092896
  similarity 11:dog 0.47633811831474304
  similarity 12:love 0.3277304768562317
11:dog
  similarity 0:the 0.13488972187042236
  similarity 1:little -0.19662168622016907
  similarity 2:pig 0.1785326451063156
  similarity 3:<eos> -0.03706585615873337
  similarity 4:every -0.577937662601471
  similarity 5:white -0.023855343461036682
  similarity 6:cat 0.18695366382598877
  similarity 7:chase 0.26908931136131287
  similarity 8:a -0.3050384819507599
  similarity 9:bite 0.1033247634768486
  similarity 10:black 0.47633811831474304
  similarity 11:dog 0.9999999403953552
  similarity 12:love 0.21171364188194275
12:love
  similarity 0:the 0.10986379534006119
  similarity 1:little 0.18874666094779968
  similarity 2:pig -0.4939063489437103
  similarity 3:<eos> 0.4061325788497925
  similarity 4:every 0.1548893004655838
  similarity 5:white 0.1925353854894638
  similarity 6:cat 0.203840434551239
  similarity 7:chase -0.12480109930038452
  similarity 8:a -0.651657223701477
  similarity 9:bite 0.13834749162197113
  similarity 10:black 0.3277304768562317
  similarity 11:dog 0.21171364188194275
  similarity 12:love 0.9999998807907104
```
