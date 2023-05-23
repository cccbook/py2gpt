# demo.py

學習排序 sorting 的範例

## Run

```
ccckmit@asus MINGW64 /d/ccc/code/py/minGPT (master)
$ python demo.py
1 -1
0 -1
1 -1
0 -1
0 -1
0 0
0 0
0 0
0 0
0 1
1 1
number of parameters: 0.09M
running on device cpu
iter_dt 0.00ms; iter 0: train loss 1.05609
iter_dt 231.01ms; iter 100: train loss 0.21134
iter_dt 318.02ms; iter 200: train loss 0.16427
iter_dt 199.01ms; iter 300: train loss 0.04406
iter_dt 247.01ms; iter 400: train loss 0.01442
iter_dt 166.01ms; iter 500: train loss 0.00518
iter_dt 169.01ms; iter 600: train loss 0.04528
iter_dt 228.01ms; iter 700: train loss 0.00880
iter_dt 251.02ms; iter 800: train loss 0.02762
iter_dt 231.01ms; iter 900: train loss 0.02851
iter_dt 168.01ms; iter 1000: train loss 0.00154
iter_dt 203.01ms; iter 1100: train loss 0.01020
iter_dt 149.01ms; iter 1200: train loss 0.01012
iter_dt 216.01ms; iter 1300: train loss 0.03334
iter_dt 248.01ms; iter 1400: train loss 0.00535
iter_dt 191.01ms; iter 1500: train loss 0.00148
iter_dt 164.01ms; iter 1600: train loss 0.01191
iter_dt 201.01ms; iter 1700: train loss 0.00732
iter_dt 204.01ms; iter 1800: train loss 0.00187
iter_dt 190.01ms; iter 1900: train loss 0.00039
train final score: 5000/5000 = 100.00% correct
test final score: 5000/5000 = 100.00% correct
input sequence  : [[0, 0, 2, 1, 0, 1]]
predicted sorted: [[0, 0, 0, 1, 1, 2]]
gt sort         : [0, 0, 0, 1, 1, 2]
matches         : True
```
