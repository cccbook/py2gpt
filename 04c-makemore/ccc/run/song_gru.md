
$ python makemore.py -i ../_corpus/song.txt -o output/song_gru --type gru
{'input_file': '../_corpus/song.txt', 'work_dir': 'output/song_gru', 'resume': False, 'sample_only':
False, 'num_workers': 4, 'max_steps': -1, 'device': 'cpu', 'seed': 3407, 'top_k': -1, 'type': 'gru',
'n_layer': 4, 'n_head': 4, 'n_embd': 64, 'n_embd2': 64, 'batch_size': 32, 'learning_rate': 0.0005, 'weight_decay': 0.01}
number of examples in the dataset: 23
max word length: 26
number of unique characters in the vocabulary: 12
vocabulary:
 $()-1234567
split up the dataset into 21 training examples and 2 test examples
dataset determined that: vocab_size=13, block_size=27
model #params: 26509
step 0 | loss 2.6273 | step time 30260.74ms
step 10 | loss 2.4645 | step time 143.01ms
step 20 | loss 2.3388 | step time 138.01ms
step 30 | loss 2.2057 | step time 145.01ms
step 40 | loss 2.1135 | step time 137.01ms
step 50 | loss 1.9808 | step time 157.01ms
step 60 | loss 1.8983 | step time 172.01ms
step 70 | loss 1.9737 | step time 164.01ms
step 80 | loss 1.7858 | step time 150.01ms
step 90 | loss 1.7709 | step time 144.01ms
step 100 | loss 1.6996 | step time 145.01ms
step 110 | loss 1.6477 | step time 125.01ms
step 120 | loss 1.6270 | step time 104.00ms
step 130 | loss 1.5397 | step time 101.01ms
step 140 | loss 1.5193 | step time 104.99ms
step 150 | loss 1.4942 | step time 134.01ms
step 160 | loss 1.4231 | step time 157.01ms
step 170 | loss 1.4345 | step time 119.01ms
step 180 | loss 1.3750 | step time 109.37ms
step 190 | loss 1.3128 | step time 109.37ms
step 200 | loss 1.2630 | step time 145.57ms
--------------------------------------------------------------------------------
2 samples that are in train:
$
$
0 samples that are in test:
8 samples that are new:
656
$1
135545 4423322245
-(1)
53 45332221233-22345631555
(1)7(1)(755
3123 54334545
1
--------------------------------------------------------------------------------
step 210 | loss 1.3043 | step time 159.01ms
step 220 | loss 1.1426 | step time 172.01ms
step 230 | loss 1.2071 | step time 99.81ms
step 240 | loss 1.1579 | step time 109.99ms
step 250 | loss 1.0936 | step time 93.75ms
step 260 | loss 0.9953 | step time 127.01ms
step 270 | loss 1.0221 | step time 128.44ms
step 280 | loss 0.9373 | step time 117.01ms
step 290 | loss 0.9319 | step time 183.60ms
step 300 | loss 0.9397 | step time 125.00ms
step 310 | loss 0.8996 | step time 122.01ms
step 320 | loss 0.8739 | step time 93.75ms
step 330 | loss 0.8225 | step time 89.85ms
step 340 | loss 0.7990 | step time 93.75ms
step 350 | loss 0.8033 | step time 103.01ms
step 360 | loss 0.7195 | step time 93.75ms
step 370 | loss 0.7139 | step time 93.75ms
step 380 | loss 0.6715 | step time 110.01ms
step 390 | loss 0.6092 | step time 92.07ms
step 400 | loss 0.6518 | step time 111.08ms
--------------------------------------------------------------------------------
5 samples that are in train:
$
$
$
$
$
0 samples that are in test:
5 samples that are new:
1231555631
3553433
65653533-221)1756656535 55
(1)65 -42
12333321 535 43 543323
--------------------------------------------------------------------------------
step 410 | loss 0.6698 | step time 129.01ms
step 420 | loss 0.6143 | step time 95.09ms
step 430 | loss 0.5513 | step time 93.75ms
step 440 | loss 0.4970 | step time 96.76ms
step 450 | loss 0.4970 | step time 93.75ms
step 460 | loss 0.5124 | step time 108.07ms
step 470 | loss 0.4977 | step time 89.33ms
step 480 | loss 0.4740 | step time 119.11ms
step 490 | loss 0.4975 | step time 93.75ms
step 500 | loss 0.4422 | step time 98.35ms
step 500 train loss: 0.45349106192588806 test loss: 1.6785510778427124
test loss 1.6785510778427124 is the best so far, saving model to output/song_gru\model.pt
step 510 | loss 0.4284 | step time 93.75ms
step 520 | loss 0.4035 | step time 93.75ms
step 530 | loss 0.4254 | step time 93.75ms
step 540 | loss 0.4195 | step time 93.75ms
step 550 | loss 0.4021 | step time 112.00ms
step 560 | loss 0.4198 | step time 82.09ms
step 570 | loss 0.3961 | step time 150.37ms
step 580 | loss 0.3580 | step time 105.01ms
step 590 | loss 0.3465 | step time 319.02ms
step 600 | loss 0.3403 | step time 109.01ms
--------------------------------------------------------------------------------
3 samples that are in train:
$
$
$
0 samples that are in test:
7 samples that are new:
1155665 44332 1231
1155665 44332-
345 43321
533 422 43321
345)
65(1)  (1)5(1)  4 33455
1175666566 135555
--------------------------------------------------------------------------------
step 610 | loss 0.3289 | step time 93.75ms
step 620 | loss 0.3173 | step time 108.34ms
step 630 | loss 0.3117 | step time 110.98ms
step 640 | loss 0.3120 | step time 110.94ms
step 650 | loss 0.3075 | step time 109.37ms
step 660 | loss 0.3240 | step time 101.23ms
step 670 | loss 0.3012 | step time 112.93ms
step 680 | loss 0.3007 | step time 109.38ms
step 690 | loss 0.3046 | step time 93.77ms
step 700 | loss 0.2867 | step time 91.33ms
step 710 | loss 0.2663 | step time 111.67ms
step 720 | loss 0.2860 | step time 94.84ms
step 730 | loss 0.2666 | step time 137.00ms
step 740 | loss 0.2573 | step time 152.01ms
step 750 | loss 0.2568 | step time 111.43ms
step 760 | loss 0.2663 | step time 109.38ms
step 770 | loss 0.2510 | step time 135.01ms
step 780 | loss 0.2586 | step time 93.75ms
step 790 | loss 0.3100 | step time 93.75ms
step 800 | loss 0.2685 | step time 93.75ms
--------------------------------------------------------------------------------
6 samples that are in train:
1123321231
$
$
$
(1)5(1)  (1)5(1)
1231 1231
0 samples that are in test:
4 samples that are new:
15534 33-
1231 123155
(1)(1)5(1)  65(1)
3(1)75 443345
--------------------------------------------------------------------------------
step 810 | loss 0.2854 | step time 100.05ms
step 820 | loss 0.2438 | step time 90.71ms
step 830 | loss 0.2467 | step time 117.00ms
step 840 | loss 0.2750 | step time 120.16ms
step 850 | loss 0.2635 | step time 115.61ms
step 860 | loss 0.2448 | step time 93.75ms
step 870 | loss 0.2359 | step time 151.01ms
step 880 | loss 0.2549 | step time 147.01ms
step 890 | loss 0.2605 | step time 109.38ms
step 900 | loss 0.2549 | step time 110.90ms
step 910 | loss 0.2389 | step time 128.01ms
step 920 | loss 0.2462 | step time 115.01ms
step 930 | loss 0.2433 | step time 109.38ms
step 940 | loss 0.2732 | step time 126.33ms
step 950 | loss 0.2219 | step time 95.35ms
step 960 | loss 0.2439 | step time 95.43ms
step 970 | loss 0.2417 | step time 101.40ms
step 980 | loss 0.2476 | step time 93.75ms
step 990 | loss 0.2232 | step time 93.75ms
step 1000 | loss 0.2393 | step time 139.30ms
step 1000 train loss: 0.23591633141040802 test loss: 2.536825656890869
--------------------------------------------------------------------------------
9 samples that are in train:
345 345
(1)7653 (1)765
$
$
533 422 1234555
(1)7653 (1)765
$
565631 565431
$
0 samples that are in test:
1 samples that are new:
111756666565355555656535
--------------------------------------------------------------------------------
step 1010 | loss 0.2471 | step time 119.01ms
step 1020 | loss 0.2548 | step time 120.01ms
step 1030 | loss 0.2369 | step time 112.00ms
step 1040 | loss 0.2197 | step time 98.65ms
step 1050 | loss 0.2232 | step time 111.01ms
step 1060 | loss 0.2373 | step time 120.98ms
step 1070 | loss 0.2179 | step time 113.81ms
step 1080 | loss 0.2264 | step time 167.01ms
step 1090 | loss 0.2091 | step time 141.70ms
step 1100 | loss 0.2246 | step time 127.79ms
step 1110 | loss 0.2316 | step time 109.38ms
step 1120 | loss 0.2114 | step time 105.94ms
step 1130 | loss 0.2388 | step time 103.04ms
step 1140 | loss 0.2543 | step time 93.73ms
step 1150 | loss 0.2431 | step time 111.01ms
step 1160 | loss 0.2391 | step time 114.01ms
step 1170 | loss 0.2297 | step time 109.00ms
step 1180 | loss 0.2488 | step time 126.00ms
step 1190 | loss 0.1964 | step time 109.01ms
step 1200 | loss 0.2128 | step time 123.01ms
--------------------------------------------------------------------------------
8 samples that are in train:
$
$
33455 43453
(1)5(1)  (1)5(1)
533 422 13551
345 345
(1)5(1)  (1)5(1)
1231 1231
0 samples that are in test:
2 samples that are new:
-$
2222234 333323-
--------------------------------------------------------------------------------
step 1210 | loss 0.2305 | step time 131.01ms
step 1220 | loss 0.2406 | step time 147.01ms
step 1230 | loss 0.2209 | step time 139.01ms
step 1240 | loss 0.2243 | step time 167.01ms
step 1250 | loss 0.2134 | step time 142.01ms
step 1260 | loss 0.2145 | step time 125.01ms
step 1270 | loss 0.2247 | step time 124.01ms
step 1280 | loss 0.2254 | step time 143.01ms
step 1290 | loss 0.2229 | step time 175.01ms
step 1300 | loss 0.2318 | step time 185.76ms
step 1310 | loss 0.2217 | step time 182.01ms
step 1320 | loss 0.2260 | step time 100.50ms
step 1330 | loss 0.2311 | step time 120.08ms
step 1340 | loss 0.2383 | step time 188.01ms
step 1350 | loss 0.2632 | step time 171.01ms
step 1360 | loss 0.2172 | step time 137.01ms
step 1370 | loss 0.2241 | step time 96.97ms
step 1380 | loss 0.2411 | step time 123.01ms
step 1390 | loss 0.2255 | step time 159.01ms
step 1400 | loss 0.2354 | step time 138.00ms
--------------------------------------------------------------------------------
8 samples that are in train:
$
5544332 5544332
345 345
1117566665653555557666656-
67(1)53 5421
33455 43453
5544332 5544332
66565353323-22122553323-
0 samples that are in test:
2 samples that are new:
(1)5(1)  (1)5(1
1117566665 443345
--------------------------------------------------------------------------------
step 1410 | loss 0.2191 | step time 109.38ms
step 1420 | loss 0.2174 | step time 110.78ms
step 1430 | loss 0.2008 | step time 93.75ms
step 1440 | loss 0.2332 | step time 110.01ms
step 1450 | loss 0.2188 | step time 118.01ms
step 1460 | loss 0.2131 | step time 145.01ms
step 1470 | loss 0.2409 | step time 125.01ms
step 1480 | loss 0.2360 | step time 103.01ms
step 1490 | loss 0.1999 | step time 118.01ms
step 1500 | loss 0.2123 | step time 115.01ms
step 1500 train loss: 0.22043651342391968 test loss: 2.983582019805908
step 1510 | loss 0.2474 | step time 127.01ms
step 1520 | loss 0.2390 | step time 176.01ms
step 1530 | loss 0.2207 | step time 169.01ms
step 1540 | loss 0.2232 | step time 164.01ms
step 1550 | loss 0.2076 | step time 147.01ms
step 1560 | loss 0.2065 | step time 148.01ms
step 1570 | loss 0.2259 | step time 121.00ms
step 1580 | loss 0.2530 | step time 148.01ms
step 1590 | loss 0.2074 | step time 155.01ms
step 1600 | loss 0.2124 | step time 149.01ms
--------------------------------------------------------------------------------
10 samples that are in train:
66565353323-22122553323-
$
$
565631 565431
$
1123321231
1117566665653555557666656-
1155665 4433221
$
$
0 samples that are in test:
0 samples that are new:
--------------------------------------------------------------------------------
step 1610 | loss 0.2171 | step time 146.01ms
step 1620 | loss 0.2730 | step time 160.01ms
step 1630 | loss 0.2171 | step time 293.02ms
step 1640 | loss 0.1887 | step time 200.01ms
step 1650 | loss 0.2299 | step time 168.01ms
step 1660 | loss 0.2283 | step time 187.01ms
step 1670 | loss 0.2252 | step time 609.04ms
step 1680 | loss 0.2120 | step time 190.01ms
step 1690 | loss 0.2427 | step time 233.01ms
step 1700 | loss 0.2460 | step time 135.00ms
step 1710 | loss 0.2296 | step time 135.01ms
step 1720 | loss 0.2075 | step time 166.01ms
step 1730 | loss 0.2112 | step time 143.01ms
step 1740 | loss 0.2010 | step time 149.01ms
step 1750 | loss 0.2209 | step time 157.01ms
step 1760 | loss 0.1956 | step time 170.01ms
step 1770 | loss 0.2161 | step time 143.01ms
step 1780 | loss 0.2372 | step time 136.59ms
step 1790 | loss 0.2065 | step time 125.52ms
step 1800 | loss 0.2266 | step time 114.85ms
--------------------------------------------------------------------------------
9 samples that are in train:
$
$
(1)5(1)  (1)5(1)
1155665 4433221
533 422 13551
1117566665653555557666656-
$
345 345
1231 1231
0 samples that are in test:
1 samples that are new:
(1)7653
--------------------------------------------------------------------------------
step 1810 | loss 0.2389 | step time 104.98ms
step 1820 | loss 0.2238 | step time 105.01ms
step 1830 | loss 0.2075 | step time 99.38ms
step 1840 | loss 0.2178 | step time 111.73ms
step 1850 | loss 0.2283 | step time 110.01ms
step 1860 | loss 0.2158 | step time 150.38ms
step 1870 | loss 0.2212 | step time 132.01ms
step 1880 | loss 0.2351 | step time 108.00ms
step 1890 | loss 0.2105 | step time 154.01ms
step 1900 | loss 0.2025 | step time 133.01ms
step 1910 | loss 0.2234 | step time 109.01ms
step 1920 | loss 0.2160 | step time 112.01ms
step 1930 | loss 0.2079 | step time 105.44ms
step 1940 | loss 0.2133 | step time 97.89ms
step 1950 | loss 0.2466 | step time 113.53ms
step 1960 | loss 0.2129 | step time 112.96ms
step 1970 | loss 0.2233 | step time 141.01ms
step 1980 | loss 0.2308 | step time 147.01ms
step 1990 | loss 0.2207 | step time 93.75ms
step 2000 | loss 0.2405 | step time 93.75ms
step 2000 train loss: 0.2174275666475296 test loss: 3.3288052082061768
--------------------------------------------------------------------------------
10 samples that are in train:
533 422 1234555
1123321231
$
5544332 5544332
1117566665653555557666656-
(1)7653 (1)765
2222234 3333345
$
(1)7653 (1)765
1123321231
0 samples that are in test:
0 samples that are new:
--------------------------------------------------------------------------------
step 2010 | loss 0.2070 | step time 117.72ms
step 2020 | loss 0.2174 | step time 130.01ms
step 2030 | loss 0.2100 | step time 99.20ms
step 2040 | loss 0.1999 | step time 114.63ms
step 2050 | loss 0.1988 | step time 93.75ms
step 2060 | loss 0.2259 | step time 104.85ms
step 2070 | loss 0.2006 | step time 93.75ms
step 2080 | loss 0.2078 | step time 125.01ms
step 2090 | loss 0.1999 | step time 94.57ms
step 2100 | loss 0.2056 | step time 308.02ms
step 2110 | loss 0.2680 | step time 123.01ms
step 2120 | loss 0.2093 | step time 130.01ms
step 2130 | loss 0.2287 | step time 137.01ms
step 2140 | loss 0.2285 | step time 107.19ms
step 2150 | loss 0.2234 | step time 122.58ms
step 2160 | loss 0.2261 | step time 117.00ms
step 2170 | loss 0.1852 | step time 105.66ms
step 2180 | loss 0.1991 | step time 108.01ms
step 2190 | loss 0.2146 | step time 98.59ms
step 2200 | loss 0.2338 | step time 98.24ms
--------------------------------------------------------------------------------
10 samples that are in train:
$
$
533 422 1234555
5544332 5544332
(1)5(1)  (1)5(1)
533 422 1234555
$
2222234 3333345
$
$
0 samples that are in test:
0 samples that are new:
--------------------------------------------------------------------------------
step 2210 | loss 0.2228 | step time 86.03ms
step 2220 | loss 0.2346 | step time 102.37ms
step 2230 | loss 0.2446 | step time 93.75ms
step 2240 | loss 0.1996 | step time 114.95ms
step 2250 | loss 0.2302 | step time 112.01ms
step 2260 | loss 0.2151 | step time 125.01ms
step 2270 | loss 0.1971 | step time 93.75ms
step 2280 | loss 0.2476 | step time 106.50ms
step 2290 | loss 0.2281 | step time 93.75ms
step 2300 | loss 0.2016 | step time 112.01ms
step 2310 | loss 0.1985 | step time 121.56ms
step 2320 | loss 0.2163 | step time 101.88ms
step 2330 | loss 0.2091 | step time 115.44ms
step 2340 | loss 0.2072 | step time 125.00ms
step 2350 | loss 0.2018 | step time 125.00ms
step 2360 | loss 0.1927 | step time 148.96ms
step 2370 | loss 0.2110 | step time 137.00ms
step 2380 | loss 0.2037 | step time 170.01ms
step 2390 | loss 0.2169 | step time 153.04ms
step 2400 | loss 0.2114 | step time 102.94ms
--------------------------------------------------------------------------------
10 samples that are in train:
1117566665653555557666656-
(1)5(1)  (1)5(1)
1117566665653555557666656-
(1)7653 (1)765
$
$
345 345
$
$
565631 565431
0 samples that are in test:
0 samples that are new:
--------------------------------------------------------------------------------
step 2410 | loss 0.2262 | step time 107.06ms
step 2420 | loss 0.2022 | step time 120.04ms
step 2430 | loss 0.1829 | step time 164.01ms
step 2440 | loss 0.2186 | step time 170.66ms
step 2450 | loss 0.1988 | step time 131.01ms
step 2460 | loss 0.2291 | step time 126.53ms
step 2470 | loss 0.1800 | step time 230.02ms
step 2480 | loss 0.2361 | step time 168.01ms
step 2490 | loss 0.2261 | step time 177.01ms
step 2500 | loss 0.1883 | step time 122.01ms
step 2500 train loss: 0.21572339534759521 test loss: 3.5042672157287598
step 2510 | loss 0.2469 | step time 269.02ms
step 2520 | loss 0.2444 | step time 186.17ms
step 2530 | loss 0.2019 | step time 385.02ms
step 2540 | loss 0.1960 | step time 98.59ms
step 2550 | loss 0.2282 | step time 109.93ms
step 2560 | loss 0.2407 | step time 96.72ms
step 2570 | loss 0.2110 | step time 133.01ms
step 2580 | loss 0.2323 | step time 119.90ms
step 2590 | loss 0.2309 | step time 109.38ms
step 2600 | loss 0.2338 | step time 172.01ms
--------------------------------------------------------------------------------
10 samples that are in train:
$
$
67(1)53 5421
2222234 3333345
(1)5(1)  (1)5(1)
67(1)53 5421
2222234 3333345
1117566665653555557666656-
5544332 5544332
(1)5(1)  (1)5(1)
0 samples that are in test:
0 samples that are new:
--------------------------------------------------------------------------------
step 2610 | loss 0.2401 | step time 231.01ms
step 2620 | loss 0.2040 | step time 132.01ms
step 2630 | loss 0.2082 | step time 109.38ms
step 2640 | loss 0.2044 | step time 100.69ms
step 2650 | loss 0.2312 | step time 152.01ms
step 2660 | loss 0.2559 | step time 200.01ms
step 2670 | loss 0.2022 | step time 177.01ms
step 2680 | loss 0.2170 | step time 170.01ms
step 2690 | loss 0.2219 | step time 157.40ms
step 2700 | loss 0.2078 | step time 109.37ms
step 2710 | loss 0.2095 | step time 109.37ms
step 2720 | loss 0.2121 | step time 86.73ms
step 2730 | loss 0.2027 | step time 111.89ms
step 2740 | loss 0.2263 | step time 122.67ms
step 2750 | loss 0.2032 | step time 125.00ms
step 2760 | loss 0.2437 | step time 109.37ms
step 2770 | loss 0.2151 | step time 93.75ms
step 2780 | loss 0.2074 | step time 93.75ms
step 2790 | loss 0.2328 | step time 109.38ms
step 2800 | loss 0.2203 | step time 94.82ms
--------------------------------------------------------------------------------
10 samples that are in train:
$
345 345
$
533 422 13551
1155665 4433221
1155665 4433221
$
33455 43453
1155665 4433221
33455 43453
0 samples that are in test:
0 samples that are new:
--------------------------------------------------------------------------------
step 2810 | loss 0.2445 | step time 99.16ms
step 2820 | loss 0.2003 | step time 111.01ms
step 2830 | loss 0.2194 | step time 111.01ms
step 2840 | loss 0.1974 | step time 126.56ms
step 2850 | loss 0.1790 | step time 129.05ms
step 2860 | loss 0.2212 | step time 97.09ms
step 2870 | loss 0.2233 | step time 115.35ms
step 2880 | loss 0.2215 | step time 93.75ms
step 2890 | loss 0.2146 | step time 109.38ms
step 2900 | loss 0.2112 | step time 93.75ms
step 2910 | loss 0.2106 | step time 93.75ms
step 2920 | loss 0.1955 | step time 109.38ms
step 2930 | loss 0.2008 | step time 123.24ms
step 2940 | loss 0.1801 | step time 93.75ms
step 2950 | loss 0.2043 | step time 99.49ms
step 2960 | loss 0.1962 | step time 96.45ms
step 2970 | loss 0.2220 | step time 158.97ms
step 2980 | loss 0.2053 | step time 95.25ms
step 2990 | loss 0.2267 | step time 89.95ms
step 3000 | loss 0.2196 | step time 95.47ms
step 3000 train loss: 0.21563231945037842 test loss: 3.6767849922180176
--------------------------------------------------------------------------------
10 samples that are in train:
533 422 13551
(1)5(1)  (1)5(1)
$
533 422 13551
1117566665653555557666656-
33455 43453
533 422 1234555
(1)5(1)  (1)5(1)
(1)7653 (1)765
(1)5(1)  (1)5(1)
0 samples that are in test:
0 samples that are new:
--------------------------------------------------------------------------------
step 3010 | loss 0.2180 | step time 104.73ms
step 3020 | loss 0.2217 | step time 99.95ms
step 3030 | loss 0.2106 | step time 93.75ms
step 3040 | loss 0.2076 | step time 109.38ms
step 3050 | loss 0.2128 | step time 107.01ms
step 3060 | loss 0.2257 | step time 109.38ms
step 3070 | loss 0.2416 | step time 93.75ms
step 3080 | loss 0.2195 | step time 93.75ms
step 3090 | loss 0.2243 | step time 93.75ms
step 3100 | loss 0.2224 | step time 128.08ms
step 3110 | loss 0.2144 | step time 109.37ms
step 3120 | loss 0.2221 | step time 109.38ms
step 3130 | loss 0.1994 | step time 109.37ms
step 3140 | loss 0.2257 | step time 93.75ms
step 3150 | loss 0.1781 | step time 108.04ms
step 3160 | loss 0.2108 | step time 92.57ms
step 3170 | loss 0.2007 | step time 95.58ms
step 3180 | loss 0.2211 | step time 109.87ms
step 3190 | loss 0.2119 | step time 93.75ms
step 3200 | loss 0.2340 | step time 126.12ms
--------------------------------------------------------------------------------
10 samples that are in train:
(1)5(1)  (1)5(1)
565631 565431
(1)5(1)  (1)5(1)
33455 43453
66565353323-22122553323-
1155665 4433221
$
5544332 5544332
$
5544332 5544332
0 samples that are in test:
0 samples that are new:
--------------------------------------------------------------------------------
step 3210 | loss 0.2021 | step time 93.75ms
step 3220 | loss 0.2172 | step time 93.73ms
step 3230 | loss 0.2225 | step time 109.36ms
step 3240 | loss 0.1944 | step time 90.58ms
step 3250 | loss 0.2319 | step time 108.01ms
step 3260 | loss 0.2314 | step time 93.75ms
step 3270 | loss 0.2238 | step time 93.75ms
step 3280 | loss 0.2232 | step time 99.61ms
step 3290 | loss 0.1940 | step time 112.38ms
step 3300 | loss 0.2316 | step time 98.14ms
step 3310 | loss 0.2185 | step time 110.07ms
step 3320 | loss 0.2092 | step time 112.19ms
step 3330 | loss 0.2261 | step time 93.59ms
Traceback (most recent call last):
  File "D:\ccc\py2gpt\04c-makemore\makemore.py", line 680, in <module>
    logits, loss = model(X, Y)
                   ^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\nn\modules\module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ccc\py2gpt\04c-makemore\makemore.py", line 332, in forward
    ht = self.cell(xt, hprev) # (b, n_embd2)
         ^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\nn\modules\module.py", line 1501, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "D:\ccc\py2gpt\04c-makemore\makemore.py", line 298, in forward
    z = F.sigmoid(self.xh_to_z(xh))
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "C:\Users\user\AppData\Local\Programs\Python\Python311\Lib\site-packages\torch\nn\functional.py", line 1968, in sigmoid
    return input.sigmoid()
           ^^^^^^^^^^^^^^^
KeyboardInterrupt