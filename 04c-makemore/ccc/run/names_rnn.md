
```
$ python makemore.py -i names.txt -o names --type rnn
{'input_file': 'names.txt', 'work_dir': 'names', 'resume': False, 'sample_only': False, 'num_workers': 4, 'max_steps': -1, 'device': 'cpu', 'seed': 3407, 'top_k': -1, 'type': 'rnn', 'n_layer': 4, 'n_head': 4, 'n_embd': 64, 'n_embd2': 64, 'batch_size': 32, 'learning_rate': 0.0005, 'weight_decay': 0.01}
number of examples in the dataset: 32033
max word length: 15
number of unique characters in the vocabulary: 26
vocabulary:
abcdefghijklmnopqrstuvwxyz
split up the dataset into 31033 training examples and 1000 test examples
dataset determined that: vocab_size=27, block_size=16
model #params: 11803
step 0 | loss 3.3456 | step time 732.28ms
step 10 | loss 3.2083 | step time 17.70ms
step 20 | loss 3.0950 | step time 31.25ms
step 30 | loss 2.9975 | step time 15.62ms
step 40 | loss 2.8010 | step time 15.62ms
step 50 | loss 2.6702 | step time 32.80ms
step 60 | loss 2.6400 | step time 22.70ms
step 70 | loss 2.7012 | step time 48.80ms
step 80 | loss 2.6538 | step time 14.17ms
step 90 | loss 2.5617 | step time 25.07ms
step 100 | loss 2.5606 | step time 31.25ms
step 110 | loss 2.4792 | step time 18.07ms
step 120 | loss 2.4289 | step time 31.00ms
step 130 | loss 2.5157 | step time 15.62ms
step 140 | loss 2.3891 | step time 31.25ms
step 150 | loss 2.3510 | step time 36.33ms
step 160 | loss 2.2964 | step time 41.26ms
step 170 | loss 2.3829 | step time 15.63ms
step 180 | loss 2.4075 | step time 40.00ms
step 190 | loss 2.3507 | step time 33.00ms
step 200 | loss 2.4258 | step time 15.62ms
--------------------------------------------------------------------------------
1 samples that are in train:
ania
0 samples that are in test:
9 samples that are new:
jacya
liavkn
dtyori
xenabnd
wa
szile
ealeen
lalza
lceve
--------------------------------------------------------------------------------
step 210 | loss 2.3480 | step time 32.00ms
step 220 | loss 2.2744 | step time 44.00ms
step 230 | loss 2.3961 | step time 31.00ms
step 240 | loss 2.3560 | step time 34.00ms
step 250 | loss 2.3733 | step time 37.00ms
step 260 | loss 2.4633 | step time 37.32ms
step 270 | loss 2.3941 | step time 40.18ms
step 280 | loss 2.3169 | step time 39.17ms
step 290 | loss 2.4490 | step time 15.63ms
step 300 | loss 2.3943 | step time 31.25ms
step 310 | loss 2.2909 | step time 40.00ms
step 320 | loss 2.2388 | step time 31.25ms
step 330 | loss 2.2997 | step time 24.71ms
step 340 | loss 2.3852 | step time 31.25ms
step 350 | loss 2.4349 | step time 31.25ms
step 360 | loss 2.2738 | step time 31.25ms
step 370 | loss 2.2205 | step time 37.81ms
step 380 | loss 2.3158 | step time 29.00ms
step 390 | loss 2.3648 | step time 31.25ms
step 400 | loss 2.3409 | step time 16.67ms
--------------------------------------------------------------------------------
1 samples that are in train:
rakan
0 samples that are in test:
9 samples that are new:
anelom
mamty
fvorukeeq
idisor
liamhadayl
feynni
kuofabi
raesta
jayfun
--------------------------------------------------------------------------------
step 410 | loss 2.2723 | step time 39.51ms
step 420 | loss 2.3423 | step time 31.25ms
step 430 | loss 2.3283 | step time 23.07ms
step 440 | loss 2.2646 | step time 31.25ms
step 450 | loss 2.3665 | step time 31.25ms
step 460 | loss 2.3317 | step time 33.67ms
step 470 | loss 2.4468 | step time 31.25ms
step 480 | loss 2.2875 | step time 31.25ms
step 490 | loss 2.2722 | step time 31.25ms
step 500 | loss 2.2783 | step time 31.25ms
step 500 train loss: 2.283903121948242 test loss: 2.276024341583252
test loss 2.276024341583252 is the best so far, saving model to names\model.pt
step 510 | loss 2.3309 | step time 31.25ms
step 520 | loss 2.3652 | step time 37.00ms
step 530 | loss 2.2361 | step time 18.71ms
step 540 | loss 2.2699 | step time 33.00ms
step 550 | loss 2.2898 | step time 33.00ms
step 560 | loss 2.3312 | step time 31.25ms
step 570 | loss 2.1806 | step time 31.25ms
step 580 | loss 2.3574 | step time 30.00ms
step 590 | loss 2.2027 | step time 28.00ms
step 600 | loss 2.2842 | step time 29.08ms
--------------------------------------------------------------------------------
1 samples that are in train:
leana
0 samples that are in test:
9 samples that are new:
barelyy
uviay
uderid
rifen
eotar
enstolya
borustane
masona
awlende
--------------------------------------------------------------------------------
step 610 | loss 2.4272 | step time 15.63ms
step 620 | loss 2.2409 | step time 16.07ms
step 630 | loss 2.3513 | step time 27.69ms
step 640 | loss 2.3428 | step time 31.00ms
step 650 | loss 2.2399 | step time 71.00ms
step 660 | loss 2.1816 | step time 31.00ms
step 670 | loss 2.3052 | step time 31.25ms
step 680 | loss 2.1911 | step time 32.00ms
step 690 | loss 2.2678 | step time 21.07ms
step 700 | loss 2.3053 | step time 27.00ms
step 710 | loss 2.3120 | step time 36.00ms
step 720 | loss 2.3176 | step time 36.00ms
step 730 | loss 2.2944 | step time 52.00ms
step 740 | loss 2.1876 | step time 48.01ms
step 750 | loss 2.2709 | step time 16.71ms
step 760 | loss 2.1800 | step time 49.11ms
step 770 | loss 2.1845 | step time 15.62ms
step 780 | loss 2.2070 | step time 31.25ms
step 790 | loss 2.1834 | step time 31.25ms
step 800 | loss 2.1932 | step time 31.25ms
--------------------------------------------------------------------------------
2 samples that are in train:
anayah
gavyn
0 samples that are in test:
8 samples that are new:
mkirhig
yereel
villinx
lilatia
jaalaca
truing
siphala
llilele
--------------------------------------------------------------------------------
step 810 | loss 2.1301 | step time 50.50ms
step 820 | loss 2.2577 | step time 31.25ms
step 830 | loss 2.2947 | step time 31.25ms
step 840 | loss 2.2200 | step time 31.25ms
step 850 | loss 2.3162 | step time 14.09ms
step 860 | loss 2.2162 | step time 31.25ms
step 870 | loss 2.1798 | step time 36.23ms
step 880 | loss 2.2793 | step time 31.25ms
step 890 | loss 2.2320 | step time 31.25ms
step 900 | loss 2.2341 | step time 29.00ms
step 910 | loss 2.1919 | step time 36.11ms
step 920 | loss 2.2791 | step time 56.19ms
step 930 | loss 2.1466 | step time 15.06ms
step 940 | loss 2.2112 | step time 31.25ms
step 950 | loss 2.2387 | step time 36.00ms
step 960 | loss 2.2529 | step time 15.63ms
step 970 | loss 2.3721 | step time 38.67ms
step 980 | loss 2.4207 | step time 31.25ms
step 990 | loss 2.1586 | step time 31.25ms
step 1000 | loss 2.1987 | step time 44.09ms
step 1000 train loss: 2.205726146697998 test loss: 2.2212343215942383
test loss 2.2212343215942383 is the best so far, saving model to names\model.pt
--------------------------------------------------------------------------------
0 samples that are in train:
0 samples that are in test:
10 samples that are new:
ejel
jono
syalai
kenee
greobra
pozsan
jarlin
jyaro
adjanna
semarie
--------------------------------------------------------------------------------
step 1010 | loss 2.1309 | step time 45.01ms
step 1020 | loss 2.2520 | step time 48.00ms
step 1030 | loss 2.3055 | step time 33.00ms
step 1040 | loss 2.2761 | step time 35.00ms
step 1050 | loss 2.3827 | step time 46.00ms
step 1060 | loss 2.2013 | step time 36.00ms
step 1070 | loss 2.2168 | step time 34.00ms
step 1080 | loss 2.4380 | step time 15.62ms
step 1090 | loss 2.3078 | step time 32.00ms
step 1100 | loss 2.3031 | step time 32.00ms
step 1110 | loss 2.1287 | step time 15.62ms
step 1120 | loss 2.1888 | step time 15.62ms
step 1130 | loss 2.2798 | step time 49.00ms
step 1140 | loss 2.2813 | step time 31.25ms
step 1150 | loss 2.2644 | step time 37.00ms
step 1160 | loss 2.2881 | step time 42.01ms
step 1170 | loss 2.2567 | step time 31.00ms
step 1180 | loss 2.0975 | step time 143.01ms
step 1190 | loss 2.2638 | step time 32.00ms
step 1200 | loss 2.2711 | step time 30.00ms
--------------------------------------------------------------------------------
1 samples that are in train:
adrian
0 samples that are in test:
9 samples that are new:
alaulanah
elianneah
iahdmira
levenn
jahaye
darayn
etttide
bejote
kdiyat
--------------------------------------------------------------------------------
step 1210 | loss 2.1520 | step time 101.00ms
step 1220 | loss 2.1543 | step time 53.00ms
step 1230 | loss 2.2658 | step time 34.00ms
step 1240 | loss 2.2306 | step time 37.00ms
step 1250 | loss 2.2319 | step time 31.00ms
step 1260 | loss 2.1141 | step time 41.00ms
step 1270 | loss 2.2065 | step time 69.00ms
step 1280 | loss 2.2165 | step time 33.00ms
step 1290 | loss 2.0736 | step time 32.00ms
step 1300 | loss 2.1691 | step time 31.25ms
step 1310 | loss 2.2890 | step time 38.69ms
step 1320 | loss 2.1713 | step time 29.00ms
step 1330 | loss 2.2529 | step time 15.63ms
step 1340 | loss 2.1694 | step time 39.00ms
step 1350 | loss 2.1842 | step time 45.00ms
step 1360 | loss 2.1724 | step time 32.00ms
step 1370 | loss 2.1538 | step time 32.00ms
step 1380 | loss 2.2764 | step time 31.25ms
step 1390 | loss 2.2493 | step time 23.69ms
step 1400 | loss 2.0712 | step time 42.00ms
--------------------------------------------------------------------------------
0 samples that are in train:
0 samples that are in test:
10 samples that are new:
nthely
edmileah
jazaid
eril
elynelu
zteli
mariste
alich
luvelja
limel
--------------------------------------------------------------------------------
step 1410 | loss 2.1547 | step time 31.25ms
step 1420 | loss 2.1771 | step time 30.00ms
step 1430 | loss 2.1437 | step time 34.00ms
step 1440 | loss 2.1348 | step time 15.63ms
step 1450 | loss 2.1177 | step time 23.07ms
step 1460 | loss 2.2325 | step time 43.00ms
step 1470 | loss 2.3311 | step time 31.25ms
step 1480 | loss 2.1532 | step time 48.00ms
step 1490 | loss 2.2219 | step time 32.00ms
step 1500 | loss 2.1086 | step time 57.64ms
step 1500 train loss: 2.199504852294922 test loss: 2.1937994956970215
test loss 2.1937994956970215 is the best so far, saving model to names\model.pt
step 1510 | loss 2.0960 | step time 33.00ms
step 1520 | loss 2.2575 | step time 31.25ms
step 1530 | loss 2.2115 | step time 15.63ms
step 1540 | loss 2.1870 | step time 31.25ms
step 1550 | loss 2.1599 | step time 15.63ms
step 1560 | loss 2.1334 | step time 35.22ms
step 1570 | loss 2.1046 | step time 31.25ms
step 1580 | loss 2.2651 | step time 31.25ms
step 1590 | loss 2.1672 | step time 31.92ms
step 1600 | loss 2.1124 | step time 31.25ms
--------------------------------------------------------------------------------
1 samples that are in train:
ravi
0 samples that are in test:
9 samples that are new:
tikpa
jafthne
taydun
phervirle
yaves
shyeliah
monel
lynan
cures
--------------------------------------------------------------------------------
step 1610 | loss 2.1312 | step time 31.25ms
step 1620 | loss 2.1988 | step time 36.58ms
step 1630 | loss 2.2015 | step time 49.16ms
step 1640 | loss 2.0785 | step time 31.25ms
step 1650 | loss 2.2995 | step time 30.00ms
step 1660 | loss 2.3782 | step time 31.27ms
step 1670 | loss 2.1969 | step time 31.27ms
step 1680 | loss 2.3488 | step time 31.25ms
step 1690 | loss 2.2379 | step time 31.25ms
step 1700 | loss 2.1979 | step time 31.25ms
step 1710 | loss 2.2674 | step time 31.25ms
step 1720 | loss 2.1124 | step time 31.25ms
step 1730 | loss 2.1691 | step time 38.72ms
step 1740 | loss 2.1764 | step time 31.25ms
step 1750 | loss 2.2645 | step time 31.25ms
step 1760 | loss 2.2871 | step time 31.25ms
step 1770 | loss 2.1926 | step time 15.62ms
step 1780 | loss 2.2106 | step time 24.10ms
step 1790 | loss 2.2717 | step time 31.25ms
step 1800 | loss 2.2846 | step time 31.25ms
--------------------------------------------------------------------------------
1 samples that are in train:
lian
0 samples that are in test:
9 samples that are new:
jokalyn
kilair
meston
loina
midonchzy
orezi
lek
jazcelynn
aaray
--------------------------------------------------------------------------------
step 1810 | loss 2.1892 | step time 31.25ms
step 1820 | loss 2.2276 | step time 31.25ms
step 1830 | loss 2.1260 | step time 15.63ms
step 1840 | loss 2.0552 | step time 22.08ms
step 1850 | loss 2.1603 | step time 15.63ms
step 1860 | loss 2.2269 | step time 31.25ms
step 1870 | loss 2.2362 | step time 31.25ms
step 1880 | loss 2.3379 | step time 31.25ms
step 1890 | loss 2.2555 | step time 31.25ms
step 1900 | loss 2.0626 | step time 15.63ms
step 1910 | loss 2.1413 | step time 31.25ms
step 1920 | loss 2.1984 | step time 31.25ms
step 1930 | loss 2.1393 | step time 31.25ms
step 1940 | loss 2.1208 | step time 67.00ms
step 1950 | loss 2.1956 | step time 62.00ms
step 1960 | loss 2.1551 | step time 31.25ms
step 1970 | loss 2.1783 | step time 31.25ms
step 1980 | loss 2.2478 | step time 35.00ms
step 1990 | loss 2.1985 | step time 31.25ms
step 2000 | loss 2.1691 | step time 35.00ms
step 2000 train loss: 2.1860857009887695 test loss: 2.1709251403808594
test loss 2.1709251403808594 is the best so far, saving model to names\model.pt
--------------------------------------------------------------------------------
1 samples that are in train:
hava
0 samples that are in test:
9 samples that are new:
kkoelie
amanel
jaja
zeure
khayus
saslen
rechor
yonniee
dakeysia
--------------------------------------------------------------------------------
step 2010 | loss 2.2222 | step time 30.00ms
step 2020 | loss 2.3598 | step time 31.25ms
step 2030 | loss 2.2535 | step time 64.00ms
step 2040 | loss 2.1919 | step time 44.00ms
step 2050 | loss 2.2088 | step time 31.00ms
step 2060 | loss 2.1491 | step time 30.00ms
step 2070 | loss 2.2815 | step time 36.00ms
step 2080 | loss 2.1562 | step time 30.00ms
step 2090 | loss 2.1626 | step time 28.99ms
step 2100 | loss 2.2591 | step time 41.00ms
step 2110 | loss 2.2088 | step time 31.25ms
step 2120 | loss 2.1914 | step time 31.25ms
step 2130 | loss 2.2590 | step time 37.00ms
step 2140 | loss 2.2901 | step time 31.25ms
step 2150 | loss 2.1513 | step time 29.00ms
step 2160 | loss 2.3280 | step time 18.24ms
step 2170 | loss 2.0537 | step time 31.25ms
step 2180 | loss 2.1276 | step time 31.25ms
step 2190 | loss 2.2250 | step time 33.00ms
step 2200 | loss 2.2685 | step time 29.00ms
--------------------------------------------------------------------------------
0 samples that are in train:
0 samples that are in test:
10 samples that are new:
lekeisli
lekeniie
izomi
qiyza
dengin
kylib
kenlon
jalyna
liadalva
amareel
--------------------------------------------------------------------------------
step 2210 | loss 2.3311 | step time 34.27ms
step 2220 | loss 2.1912 | step time 17.07ms
step 2230 | loss 2.0613 | step time 13.08ms
step 2240 | loss 2.1604 | step time 29.00ms
step 2250 | loss 2.2245 | step time 31.25ms
step 2260 | loss 2.1380 | step time 32.78ms
step 2270 | loss 2.3685 | step time 41.57ms
step 2280 | loss 2.0516 | step time 21.08ms
step 2290 | loss 2.1480 | step time 31.25ms
step 2300 | loss 2.1784 | step time 31.25ms
step 2310 | loss 2.2535 | step time 46.88ms
step 2320 | loss 2.1072 | step time 31.25ms
step 2330 | loss 2.0592 | step time 31.25ms
step 2340 | loss 2.0972 | step time 46.71ms
step 2350 | loss 2.1233 | step time 36.00ms
step 2360 | loss 2.3277 | step time 34.45ms
step 2370 | loss 2.1239 | step time 24.72ms
step 2380 | loss 2.2450 | step time 31.25ms
step 2390 | loss 2.2690 | step time 35.00ms
step 2400 | loss 2.2276 | step time 31.25ms
--------------------------------------------------------------------------------
1 samples that are in train:
jora
0 samples that are in test:
9 samples that are new:
reyut
ivina
bresaan
kahdi
anheriel
btumitan
janot
trifirah
kanisse
--------------------------------------------------------------------------------
step 2410 | loss 2.2340 | step time 31.25ms
step 2420 | loss 2.1439 | step time 31.25ms
step 2430 | loss 2.2053 | step time 33.00ms
step 2440 | loss 2.1984 | step time 31.25ms
step 2450 | loss 2.1854 | step time 31.25ms
step 2460 | loss 2.1835 | step time 31.25ms
step 2470 | loss 2.0523 | step time 30.00ms
step 2480 | loss 2.0193 | step time 15.63ms
step 2490 | loss 2.2276 | step time 31.25ms
step 2500 | loss 2.1527 | step time 33.13ms
step 2500 train loss: 2.1644628047943115 test loss: 2.158691883087158
test loss 2.158691883087158 is the best so far, saving model to names\model.pt
step 2510 | loss 2.1647 | step time 32.00ms
step 2520 | loss 2.0482 | step time 29.00ms
step 2530 | loss 2.1391 | step time 31.25ms
step 2540 | loss 2.0742 | step time 31.25ms
step 2550 | loss 2.0931 | step time 48.00ms
step 2560 | loss 2.0809 | step time 31.25ms
step 2570 | loss 2.1221 | step time 20.06ms
step 2580 | loss 2.1892 | step time 22.49ms
step 2590 | loss 2.1331 | step time 31.25ms
step 2600 | loss 2.1302 | step time 16.08ms
--------------------------------------------------------------------------------
4 samples that are in train:
raven
syla
kayley
sasha
0 samples that are in test:
6 samples that are new:
pezlit
sriell
trenbryn
jevus
lasa
shamard
--------------------------------------------------------------------------------
step 2610 | loss 2.1468 | step time 31.25ms
step 2620 | loss 2.1717 | step time 31.25ms
step 2630 | loss 2.1011 | step time 36.18ms
step 2640 | loss 2.2023 | step time 31.00ms
step 2650 | loss 2.1518 | step time 28.00ms
step 2660 | loss 2.1255 | step time 21.69ms
step 2670 | loss 2.1804 | step time 38.25ms
step 2680 | loss 2.3262 | step time 29.00ms
step 2690 | loss 2.2042 | step time 15.63ms
step 2700 | loss 2.0850 | step time 31.25ms
step 2710 | loss 2.2185 | step time 31.25ms
step 2720 | loss 2.1634 | step time 15.63ms
step 2730 | loss 2.2542 | step time 21.70ms
step 2740 | loss 2.0746 | step time 32.00ms
step 2750 | loss 2.0937 | step time 32.00ms
step 2760 | loss 2.1851 | step time 31.25ms
step 2770 | loss 2.0664 | step time 46.80ms
step 2780 | loss 2.4471 | step time 31.00ms
step 2790 | loss 2.1617 | step time 29.00ms
step 2800 | loss 2.0722 | step time 31.25ms
--------------------------------------------------------------------------------
2 samples that are in train:
maliya
jazlyn
0 samples that are in test:
8 samples that are new:
sarishi
arder
monistann
yud
thkiy
shifhar
raydanra
tyo
--------------------------------------------------------------------------------
step 2810 | loss 2.1187 | step time 77.01ms
step 2820 | loss 2.2651 | step time 33.00ms
step 2830 | loss 2.2563 | step time 31.00ms
step 2840 | loss 2.2445 | step time 31.92ms
step 2850 | loss 2.2043 | step time 15.63ms
step 2860 | loss 2.2769 | step time 15.62ms
step 2870 | loss 2.2677 | step time 32.97ms
step 2880 | loss 2.2572 | step time 42.09ms
step 2890 | loss 2.1665 | step time 31.23ms
step 2900 | loss 2.0888 | step time 30.00ms
step 2910 | loss 2.1525 | step time 15.63ms
step 2920 | loss 2.1078 | step time 26.15ms
step 2930 | loss 2.1021 | step time 31.25ms
step 2940 | loss 2.0674 | step time 15.63ms
step 2950 | loss 2.1527 | step time 43.18ms
step 2960 | loss 2.2446 | step time 29.00ms
step 2970 | loss 2.0724 | step time 31.25ms
step 2980 | loss 2.2274 | step time 46.00ms
step 2990 | loss 2.1432 | step time 31.25ms
step 3000 | loss 2.1259 | step time 31.25ms
step 3000 train loss: 2.167146682739258 test loss: 2.145080327987671
test loss 2.145080327987671 is the best so far, saving model to names\model.pt
--------------------------------------------------------------------------------
0 samples that are in train:
0 samples that are in test:
10 samples that are new:
enanti
jadarie
ablaein
lionn
vinjon
meriosk
siyafrush
adriz
denkda
ebisga
--------------------------------------------------------------------------------
step 3010 | loss 2.0574 | step time 31.25ms
step 3020 | loss 2.0345 | step time 31.25ms
```