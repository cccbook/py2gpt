
$ python makemore.py -i ../_corpus/song_star.txt -o output/song_star_gru --type gru
{'input_file': '../_corpus/song_star.txt', 'work_dir': 'output/song_star_gru', 'resume': False, 'sample_only': False, 'num_workers': 4, 'max_steps': -1, 'device': 'cpu', 'seed': 3407, 'top_k': -1, 'type': 'gru', 'n_layer': 4, 'n_head': 4, 'n_embd': 64, 'n_embd2': 64, 'batch_size': 32, 'learning_rate':
0.0005, 'weight_decay': 0.01}
number of examples in the dataset: 69
max word length: 15
number of unique characters in the vocabulary: 7
vocabulary:
 123456
split up the dataset into 63 training examples and 6 test examples
dataset determined that: vocab_size=8, block_size=16
model #params: 25864
step 0 | loss 2.0917 | step time 28520.64ms
step 10 | loss 1.8558 | step time 92.01ms
step 20 | loss 1.6443 | step time 93.01ms
step 30 | loss 1.4454 | step time 89.02ms
step 40 | loss 1.2595 | step time 77.00ms
step 50 | loss 1.0918 | step time 74.00ms
step 60 | loss 0.9492 | step time 81.00ms
step 70 | loss 0.8459 | step time 82.01ms
step 80 | loss 0.7408 | step time 72.99ms
step 90 | loss 0.6633 | step time 111.01ms
step 100 | loss 0.5842 | step time 197.01ms
step 110 | loss 0.5134 | step time 89.00ms
step 120 | loss 0.4528 | step time 74.01ms
step 130 | loss 0.4048 | step time 87.00ms
step 140 | loss 0.3641 | step time 90.01ms
step 150 | loss 0.3329 | step time 82.00ms
step 160 | loss 0.3029 | step time 114.01ms
step 170 | loss 0.2632 | step time 70.00ms
step 180 | loss 0.2585 | step time 173.01ms
step 190 | loss 0.2299 | step time 100.01ms
step 200 | loss 0.2350 | step time 95.01ms
--------------------------------------------------------------------------------
0 samples that are in train:
0 samples that are in test:
10 samples that are new:
1155665544332
554433221
1155665 44332
555433221
554433221
11
55443221
165 4433221
 54433221
1155665 44332
--------------------------------------------------------------------------------
step 210 | loss 0.1955 | step time 72.01ms
step 220 | loss 0.1871 | step time 71.00ms
step 230 | loss 0.1843 | step time 75.00ms
step 240 | loss 0.1576 | step time 70.01ms
step 250 | loss 0.1578 | step time 79.00ms
step 260 | loss 0.1412 | step time 92.01ms
step 270 | loss 0.1428 | step time 76.00ms
step 280 | loss 0.1285 | step time 71.00ms
step 290 | loss 0.1249 | step time 74.01ms
step 300 | loss 0.1267 | step time 87.00ms
step 310 | loss 0.1282 | step time 72.00ms
step 320 | loss 0.1183 | step time 73.00ms
step 330 | loss 0.1006 | step time 72.01ms
step 340 | loss 0.0962 | step time 70.00ms
step 350 | loss 0.0967 | step time 76.00ms
step 360 | loss 0.0753 | step time 81.00ms
step 370 | loss 0.0903 | step time 71.00ms
step 380 | loss 0.1015 | step time 84.00ms
step 390 | loss 0.0972 | step time 90.01ms
step 400 | loss 0.0832 | step time 85.00ms
--------------------------------------------------------------------------------
7 samples that are in train:
1155665 4433221
5544332 5544332
1155665 4433221
1155665 4433221
1155665 4433221
1155665 4433221
1155665 4433221
0 samples that are in test:
3 samples that are new:
55441332
155665 4433221
554433221
--------------------------------------------------------------------------------
step 410 | loss 0.0674 | step time 84.00ms
step 420 | loss 0.0634 | step time 81.00ms
step 430 | loss 0.0686 | step time 82.00ms
step 440 | loss 0.0751 | step time 72.01ms
step 450 | loss 0.0617 | step time 74.01ms
step 460 | loss 0.0675 | step time 71.00ms
step 470 | loss 0.0513 | step time 71.00ms
step 480 | loss 0.0463 | step time 85.01ms
step 490 | loss 0.0570 | step time 73.00ms
step 500 | loss 0.0499 | step time 75.00ms
step 500 train loss: 0.051854413002729416 test loss: 0.04141395539045334
test loss 0.04141395539045334 is the best so far, saving model to output/song_star_gru\model.pt
step 510 | loss 0.0408 | step time 70.00ms
step 520 | loss 0.0432 | step time 84.00ms
step 530 | loss 0.0490 | step time 75.00ms
step 540 | loss 0.0483 | step time 79.00ms
step 550 | loss 0.0406 | step time 89.01ms
step 560 | loss 0.0440 | step time 72.00ms
step 570 | loss 0.0452 | step time 72.00ms
step 580 | loss 0.0439 | step time 72.01ms
step 590 | loss 0.0445 | step time 81.00ms
step 600 | loss 0.0521 | step time 71.00ms
--------------------------------------------------------------------------------
9 samples that are in train:
5544332 5544332
1155665 4433221
1155665 4433221
1155665 4433221
1155665 4433221
1155665 4433221
1155665 4433221
1155665 4433221
1155665 4433221
0 samples that are in test:
1 samples that are new:
1155665 4434322
--------------------------------------------------------------------------------
step 610 | loss 0.0511 | step time 73.00ms
step 620 | loss 0.0357 | step time 73.00ms
step 630 | loss 0.0495 | step time 72.00ms
step 640 | loss 0.0456 | step time 79.00ms
step 650 | loss 0.0473 | step time 76.01ms
step 660 | loss 0.0439 | step time 87.01ms
step 670 | loss 0.0520 | step time 72.00ms
step 680 | loss 0.0421 | step time 71.00ms
step 690 | loss 0.0445 | step time 72.00ms
step 700 | loss 0.0443 | step time 72.01ms
step 710 | loss 0.0459 | step time 134.01ms
step 720 | loss 0.0415 | step time 71.00ms
step 730 | loss 0.0448 | step time 74.00ms
step 740 | loss 0.0439 | step time 79.00ms
step 750 | loss 0.0464 | step time 72.00ms
step 760 | loss 0.0488 | step time 80.00ms
step 770 | loss 0.0423 | step time 77.00ms
step 780 | loss 0.0444 | step time 72.01ms
step 790 | loss 0.0423 | step time 69.00ms
step 800 | loss 0.0453 | step time 70.00ms
--------------------------------------------------------------------------------
10 samples that are in train:
5544332 5544332
1155665 4433221
1155665 4433221
5544332 5544332
1155665 4433221
1155665 4433221
1155665 4433221
1155665 4433221
5544332 5544332
1155665 4433221
0 samples that are in test:
0 samples that are new:
--------------------------------------------------------------------------------
step 810 | loss 0.0432 | step time 83.00ms
step 820 | loss 0.0433 | step time 73.00ms
step 830 | loss 0.0359 | step time 73.00ms
step 840 | loss 0.0418 | step time 72.00ms
step 850 | loss 0.0339 | step time 70.99ms
step 860 | loss 0.0448 | step time 77.01ms
step 870 | loss 0.0436 | step time 73.00ms
step 880 | loss 0.0417 | step time 80.00ms
step 890 | loss 0.0416 | step time 72.00ms
step 900 | loss 0.0471 | step time 73.00ms
step 910 | loss 0.0406 | step time 74.00ms
step 920 | loss 0.0483 | step time 77.00ms
step 930 | loss 0.0429 | step time 73.00ms
step 940 | loss 0.0413 | step time 73.00ms
step 950 | loss 0.0445 | step time 74.01ms
step 960 | loss 0.0395 | step time 87.00ms
step 970 | loss 0.0366 | step time 73.00ms
step 980 | loss 0.0466 | step time 72.00ms
step 990 | loss 0.0403 | step time 79.01ms
step 1000 | loss 0.0421 | step time 77.01ms
step 1000 train loss: 0.04134760797023773 test loss: 0.0359698049724102
test loss 0.0359698049724102 is the best so far, saving model to output/song_star_gru\model.pt
--------------------------------------------------------------------------------
10 samples that are in train:
1155665 4433221
1155665 4433221
1155665 4433221
1155665 4433221
5544332 5544332
5544332 5544332
5544332 5544332
1155665 4433221
5544332 5544332
1155665 4433221
0 samples that are in test:
0 samples that are new:
--------------------------------------------------------------------------------
step 1010 | loss 0.0412 | step time 74.00ms
step 1020 | loss 0.0389 | step time 75.00ms
step 1030 | loss 0.0461 | step time 71.00ms
step 1040 | loss 0.0362 | step time 75.01ms
Traceback (most recent call last):
  File "D:\ccc\py2gpt\04c-makemore\makemore.py", line 684, in <module>
    loss.backward()
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\_tensor.py", line 487, in backward
    torch.autograd.backward(
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\autograd\__init__.py", line 200, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
KeyboardInterrupt