# 

```
$ python makemore.py -i corpus/english.txt -o output/english --type rnn
{'input_file': 'corpus/english.txt', 'work_dir': 'output/english', 'resume': False, 'sample_only': False, 'num_workers': 4, 'max_steps': -1, 'device': 'cpu', 'seed': 3407, 'top_k': -1, 'type': 'rnn', 'n_layer': 4, 'n_head': 4, 'n_embd': 64, 'n_embd2': 64, 'batch_size': 32, 'learning_rate': 0.0005, 'weight_decay': 0.01}
number of examples in the dataset: 1000
max word length: 39
number of unique characters in the vocabulary: 19
vocabulary:
 abcdeghikloprstvwy
split up the dataset into 900 training examples and 100 test examples
dataset determined that: vocab_size=20, block_size=40
model #params: 10900
step 0 | loss 2.9803 | step time 98.46ms
step 10 | loss 2.7223 | step time 77.00ms
step 20 | loss 2.5030 | step time 62.50ms
step 30 | loss 2.2837 | step time 62.50ms
step 40 | loss 2.0792 | step time 53.71ms
step 50 | loss 1.8863 | step time 69.70ms
step 60 | loss 1.7004 | step time 102.94ms
step 70 | loss 1.5702 | step time 62.50ms
step 80 | loss 1.4051 | step time 206.01ms
step 90 | loss 1.2923 | step time 66.89ms
step 100 | loss 1.1989 | step time 77.00ms
step 110 | loss 1.0947 | step time 64.46ms
step 120 | loss 1.0362 | step time 52.99ms
step 130 | loss 0.9398 | step time 63.12ms
step 140 | loss 0.9215 | step time 65.48ms
step 150 | loss 0.8464 | step time 69.42ms
step 160 | loss 0.7740 | step time 51.34ms
step 170 | loss 0.7383 | step time 62.50ms
step 180 | loss 0.6973 | step time 97.00ms
step 190 | loss 0.6900 | step time 75.39ms
step 200 | loss 0.6575 | step time 163.94ms
--------------------------------------------------------------------------------
0 samples that are in train:
0 samples that are in test:
10 samples that are new:
thadog
evdy cat
a black phioe gite cat love cat
e tht blackthipevery caeva cathe a dog
a dogcat log cat
hry dog
t dog catg doe bittle cay blackovery bl
every catth love y catle pig
eve tha pevery phe aktle white ack cat
the lherysery catcog white a itg cat br
--------------------------------------------------------------------------------
step 210 | loss 0.5889 | step time 219.01ms
step 220 | loss 0.6168 | step time 89.27ms
step 230 | loss 0.5798 | step time 91.01ms
step 240 | loss 0.5385 | step time 75.51ms
step 250 | loss 0.5231 | step time 80.88ms
step 260 | loss 0.5053 | step time 76.67ms
step 270 | loss 0.5022 | step time 148.01ms
step 280 | loss 0.4889 | step time 163.01ms
step 290 | loss 0.4693 | step time 96.01ms
step 300 | loss 0.4520 | step time 62.50ms
step 310 | loss 0.4553 | step time 62.50ms
step 320 | loss 0.4532 | step time 75.00ms
step 330 | loss 0.4516 | step time 62.50ms
step 340 | loss 0.4282 | step time 59.71ms
step 350 | loss 0.4296 | step time 62.50ms
step 360 | loss 0.4086 | step time 89.74ms
step 370 | loss 0.4282 | step time 91.63ms
step 380 | loss 0.4033 | step time 78.12ms
step 390 | loss 0.3869 | step time 87.56ms
step 400 | loss 0.3997 | step time 84.76ms
--------------------------------------------------------------------------------
2 samples that are in train:
every dog
a white dog
0 samples that are in test:
8 samples that are new:
thi whitevery cktke lig
every cat love a pig
the littte pig
every awdle whe dog
a white  liot
the piog chase ery while cat
a black dog lore pig bha black pig blac
thi blvcack pig chase the cat
--------------------------------------------------------------------------------
step 410 | loss 0.3949 | step time 83.09ms
step 420 | loss 0.3891 | step time 84.12ms
step 430 | loss 0.3878 | step time 87.43ms
step 440 | loss 0.3831 | step time 93.75ms
step 450 | loss 0.3944 | step time 75.36ms
step 460 | loss 0.3848 | step time 101.68ms
step 470 | loss 0.3850 | step time 78.97ms
step 480 | loss 0.3500 | step time 78.12ms
step 490 | loss 0.3761 | step time 84.32ms
step 500 | loss 0.3709 | step time 81.90ms
step 500 train loss: 0.3734271824359894 test loss: 0.3732062578201294
test loss 0.3732062578201294 is the best so far, saving model to output/english\model.pt
step 510 | loss 0.3679 | step time 131.25ms
step 520 | loss 0.3661 | step time 81.25ms
step 530 | loss 0.3696 | step time 62.50ms
step 540 | loss 0.3672 | step time 72.36ms
step 550 | loss 0.3652 | step time 78.12ms
step 560 | loss 0.3646 | step time 64.39ms
step 570 | loss 0.3692 | step time 70.27ms
step 580 | loss 0.3545 | step time 62.50ms
step 590 | loss 0.3529 | step time 69.87ms
step 600 | loss 0.3596 | step time 69.75ms
--------------------------------------------------------------------------------
5 samples that are in train:
a pig chase a little pig
a cat
a pig
a white cat
a cat
1 samples that are in test:
the little dog bite the pig
4 samples that are new:
e the cat
every cat love a cat love the cat
the cat bittle pig love black pig
a black cat bite a white dog
--------------------------------------------------------------------------------
step 610 | loss 0.3712 | step time 74.01ms
step 620 | loss 0.3531 | step time 91.22ms
step 630 | loss 0.3402 | step time 99.00ms
step 640 | loss 0.3660 | step time 95.88ms
step 650 | loss 0.3574 | step time 75.18ms
step 660 | loss 0.3734 | step time 62.50ms
step 670 | loss 0.3735 | step time 62.50ms
step 680 | loss 0.3529 | step time 72.00ms
step 690 | loss 0.3703 | step time 62.50ms
step 700 | loss 0.3508 | step time 62.50ms
step 710 | loss 0.3535 | step time 72.42ms
step 720 | loss 0.3726 | step time 62.50ms
step 730 | loss 0.3751 | step time 114.24ms
step 740 | loss 0.3555 | step time 92.59ms
step 750 | loss 0.3465 | step time 63.32ms
step 760 | loss 0.3521 | step time 64.77ms
step 770 | loss 0.3533 | step time 62.50ms
step 780 | loss 0.3633 | step time 75.49ms
step 790 | loss 0.3542 | step time 63.94ms
step 800 | loss 0.3603 | step time 62.50ms
--------------------------------------------------------------------------------
5 samples that are in train:
a black dog
the black cat
the cat
every pig
every dog
0 samples that are in test:
5 samples that are new:
every little pig bite a black pig bite
the pig bite a cat
the black dog chase a dog chase the dog
a little cat loc
the pig love the white cat love the dog
--------------------------------------------------------------------------------
step 810 | loss 0.3456 | step time 62.50ms
step 820 | loss 0.3530 | step time 58.36ms
step 830 | loss 0.3536 | step time 62.50ms
step 840 | loss 0.3336 | step time 47.33ms
step 850 | loss 0.3451 | step time 80.48ms
step 860 | loss 0.3433 | step time 83.09ms
step 870 | loss 0.3371 | step time 106.24ms
step 880 | loss 0.3678 | step time 78.13ms
step 890 | loss 0.3526 | step time 62.50ms
step 900 | loss 0.3487 | step time 60.34ms
step 910 | loss 0.3628 | step time 78.12ms
step 920 | loss 0.3461 | step time 70.79ms
step 930 | loss 0.3487 | step time 61.94ms
step 940 | loss 0.3480 | step time 62.50ms
step 950 | loss 0.3638 | step time 84.02ms
step 960 | loss 0.3587 | step time 68.89ms
step 970 | loss 0.3454 | step time 63.73ms
step 980 | loss 0.3550 | step time 53.34ms
step 990 | loss 0.3555 | step time 79.11ms
step 1000 | loss 0.3590 | step time 251.01ms
step 1000 train loss: 0.34934353828430176 test loss: 0.3493626117706299
test loss 0.3493626117706299 is the best so far, saving model to output/english\model.pt
--------------------------------------------------------------------------------
5 samples that are in train:
a little dog
every white dog
a cat
every little pig love every cat
every cat love the cat
0 samples that are in test:
5 samples that are new:
pig chase a black dog
the cat bite a pig
a white pig love every white pig
a white cat love a black pig
every cat love the white cat
--------------------------------------------------------------------------------
step 1010 | loss 0.3655 | step time 68.00ms
step 1020 | loss 0.3575 | step time 104.00ms
step 1030 | loss 0.3528 | step time 66.99ms
step 1040 | loss 0.3430 | step time 71.00ms
step 1050 | loss 0.3425 | step time 103.01ms
step 1060 | loss 0.3361 | step time 94.01ms
step 1070 | loss 0.3541 | step time 73.00ms
step 1080 | loss 0.3422 | step time 68.00ms
step 1090 | loss 0.3517 | step time 69.00ms
step 1100 | loss 0.3454 | step time 87.00ms
step 1110 | loss 0.3327 | step time 68.00ms
step 1120 | loss 0.3398 | step time 74.58ms
step 1130 | loss 0.3490 | step time 67.00ms
step 1140 | loss 0.3476 | step time 68.65ms
step 1150 | loss 0.3296 | step time 71.02ms
step 1160 | loss 0.3525 | step time 65.00ms
step 1170 | loss 0.3740 | step time 69.00ms
step 1180 | loss 0.3628 | step time 66.00ms
step 1190 | loss 0.3548 | step time 69.00ms
step 1200 | loss 0.3318 | step time 81.01ms
--------------------------------------------------------------------------------
7 samples that are in train:
the black cat
a little dog
the black cat
the dog
a cat
every dog
every white dog
0 samples that are in test:
3 samples that are new:
a white pig bite a white cat chase ever
the black dog love a cat chase the pig
a black cat bite a dog
--------------------------------------------------------------------------------
step 1210 | loss 0.3519 | step time 97.07ms
step 1220 | loss 0.3337 | step time 74.01ms
step 1230 | loss 0.3385 | step time 67.00ms
step 1240 | loss 0.3458 | step time 73.01ms
step 1250 | loss 0.3408 | step time 66.00ms
step 1260 | loss 0.3497 | step time 67.48ms
step 1270 | loss 0.3492 | step time 66.28ms
step 1280 | loss 0.3516 | step time 71.00ms
step 1290 | loss 0.3632 | step time 65.01ms
step 1300 | loss 0.3406 | step time 67.00ms
step 1310 | loss 0.3432 | step time 64.00ms
step 1320 | loss 0.3479 | step time 68.00ms
step 1330 | loss 0.3499 | step time 124.66ms
step 1340 | loss 0.3380 | step time 78.01ms
step 1350 | loss 0.3458 | step time 80.26ms
step 1360 | loss 0.3423 | step time 70.88ms
step 1370 | loss 0.3511 | step time 65.25ms
step 1380 | loss 0.3546 | step time 46.87ms
step 1390 | loss 0.3629 | step time 62.50ms
step 1400 | loss 0.3467 | step time 62.50ms
--------------------------------------------------------------------------------
9 samples that are in train:
the dog
a dog
every white dog
the cat
every black cat bite a cat
a dog
a little dog
a pig
the black dog
0 samples that are in test:
1 samples that are new:
every pig chase a white cat chase the w
--------------------------------------------------------------------------------
step 1410 | loss 0.3432 | step time 53.33ms
step 1420 | loss 0.3511 | step time 62.50ms
step 1430 | loss 0.3616 | step time 104.00ms
Traceback (most recent call last):
  File "D:\ccc\py2gpt\04c-makemore\makemore.py", line 684, in <module>
    loss.backward()
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\_tensor.py", line 487, in backward
    torch.autograd.backward(
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\autograd\__init__.py", line 200, in backward
    Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the
backward pass
KeyboardInterrupt
```