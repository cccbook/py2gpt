$ python makemore.py -i ../_corpus/song_star.txt -o output/song_star --type rnn
{'input_file': '../_corpus/song_star.txt', 'work_dir': 'output/song_star', 'resume': False, 'sample_only': False, 'num_workers': 4, 'max_steps': -1, 'device': 'cpu', 'seed': 3407, 'top_k': -1, 'type': 'rnn', 'n_layer': 4, 'n_head': 4, 'n_embd': 64, 'n_embd2': 64, 'batch_size': 32, 'learning_rate': 0.0005, 'weight_decay': 0.01}
number of examples in the dataset: 80
max word length: 15
number of unique characters in the vocabulary: 8
vocabulary:
 $123456
split up the dataset into 72 training examples and 8 test examples
dataset determined that: vocab_size=9, block_size=16
model #params: 9481
step 0 | loss 2.2909 | step time 27581.58ms
step 10 | loss 1.9677 | step time 36.00ms
step 20 | loss 1.6874 | step time 35.00ms
step 30 | loss 1.4466 | step time 34.00ms
step 40 | loss 1.2407 | step time 34.00ms
step 50 | loss 1.0845 | step time 32.00ms
step 60 | loss 0.9122 | step time 33.00ms
step 70 | loss 0.7799 | step time 33.00ms
step 80 | loss 0.7440 | step time 35.00ms
step 90 | loss 0.6099 | step time 31.00ms
step 100 | loss 0.5718 | step time 33.00ms
step 110 | loss 0.4973 | step time 33.00ms
step 120 | loss 0.4475 | step time 36.01ms
step 130 | loss 0.3739 | step time 34.00ms
step 140 | loss 0.3963 | step time 34.00ms
step 150 | loss 0.3361 | step time 32.00ms
step 160 | loss 0.3160 | step time 48.00ms
step 170 | loss 0.2935 | step time 33.00ms
step 180 | loss 0.2623 | step time 32.00ms
step 190 | loss 0.2542 | step time 37.00ms
step 200 | loss 0.2194 | step time 32.00ms
--------------------------------------------------------------------------------
4 samples that are in train:
1155665 4433221
$
5544332 5544332
1155665 4433221
0 samples that are in test:
6 samples that are new:
1155665 443322
554433221
155665 4433221
1155665665 4433
5544332 5543322
55544332 554433
--------------------------------------------------------------------------------
step 210 | loss 0.2255 | step time 32.00ms
step 220 | loss 0.2352 | step time 36.00ms
step 230 | loss 0.2332 | step time 33.00ms
step 240 | loss 0.2279 | step time 33.00ms
step 250 | loss 0.2158 | step time 32.99ms
step 260 | loss 0.2538 | step time 33.00ms
step 270 | loss 0.2055 | step time 33.00ms
step 280 | loss 0.1951 | step time 33.00ms
step 290 | loss 0.2036 | step time 40.00ms
step 300 | loss 0.2041 | step time 64.00ms
step 310 | loss 0.2071 | step time 35.00ms
step 320 | loss 0.1923 | step time 33.02ms
step 330 | loss 0.1912 | step time 33.00ms
step 340 | loss 0.1707 | step time 33.00ms
step 350 | loss 0.1512 | step time 32.00ms
step 360 | loss 0.1873 | step time 37.00ms
step 370 | loss 0.1933 | step time 34.02ms
step 380 | loss 0.1465 | step time 32.00ms
step 390 | loss 0.1292 | step time 35.00ms
step 400 | loss 0.1629 | step time 33.00ms
--------------------------------------------------------------------------------
7 samples that are in train:
1155665 4433221
1155665 4433221
$
1155665 4433221
1155665 4433221
1155665 4433221
$
0 samples that are in test:
3 samples that are new:
11
1
554433221
--------------------------------------------------------------------------------
step 410 | loss 0.1580 | step time 33.00ms
step 420 | loss 0.1305 | step time 39.00ms
step 430 | loss 0.1136 | step time 33.00ms
step 440 | loss 0.1124 | step time 33.00ms
step 450 | loss 0.0871 | step time 33.00ms
step 460 | loss 0.1224 | step time 34.00ms
step 470 | loss 0.0995 | step time 34.00ms
step 480 | loss 0.0933 | step time 35.00ms
step 490 | loss 0.0937 | step time 34.00ms
step 500 | loss 0.1196 | step time 36.00ms
step 500 train loss: 0.1002805307507515 test loss: 0.09006018191576004
test loss 0.09006018191576004 is the best so far, saving model to output/song_star\model.pt
step 510 | loss 0.1227 | step time 35.00ms
step 520 | loss 0.1014 | step time 34.00ms
step 530 | loss 0.0956 | step time 64.01ms
step 540 | loss 0.1057 | step time 68.00ms
step 550 | loss 0.0945 | step time 35.00ms
step 560 | loss 0.1100 | step time 37.00ms
step 570 | loss 0.0950 | step time 32.00ms
step 580 | loss 0.1037 | step time 34.00ms
step 590 | loss 0.0812 | step time 35.00ms
step 600 | loss 0.0839 | step time 38.00ms
--------------------------------------------------------------------------------
10 samples that are in train:
1155665 4433221
1155665 4433221
1155665 4433221
1155665 4433221
1155665 4433221
$
$
1155665 4433221
$
5544332 5544332
0 samples that are in test:
0 samples that are new:
--------------------------------------------------------------------------------
step 610 | loss 0.1026 | step time 34.00ms
step 620 | loss 0.0793 | step time 33.00ms
step 630 | loss 0.0803 | step time 33.00ms
step 640 | loss 0.1106 | step time 35.00ms
step 650 | loss 0.0967 | step time 34.02ms
step 660 | loss 0.0871 | step time 33.00ms
step 670 | loss 0.0936 | step time 32.00ms
step 680 | loss 0.0981 | step time 39.00ms
step 690 | loss 0.0792 | step time 36.00ms
step 700 | loss 0.0729 | step time 33.00ms
step 710 | loss 0.0797 | step time 35.00ms
step 720 | loss 0.0755 | step time 34.00ms
step 730 | loss 0.0838 | step time 31.00ms
step 740 | loss 0.1070 | step time 35.00ms
step 750 | loss 0.0748 | step time 33.00ms
step 760 | loss 0.0995 | step time 33.00ms
step 770 | loss 0.0938 | step time 34.00ms
step 780 | loss 0.0822 | step time 34.00ms
step 790 | loss 0.0791 | step time 34.00ms
step 800 | loss 0.0863 | step time 32.00ms
--------------------------------------------------------------------------------
10 samples that are in train:
1155665 4433221
5544332 5544332
$
$
1155665 4433221
5544332 5544332
5544332 5544332
$
1155665 4433221
1155665 4433221
0 samples that are in test:
0 samples that are new:
--------------------------------------------------------------------------------
step 810 | loss 0.0928 | step time 34.00ms
step 820 | loss 0.0697 | step time 39.00ms
step 830 | loss 0.0906 | step time 38.00ms
step 840 | loss 0.0870 | step time 34.02ms
step 850 | loss 0.1033 | step time 34.00ms
step 860 | loss 0.0889 | step time 33.00ms
step 870 | loss 0.0813 | step time 36.00ms
step 880 | loss 0.0853 | step time 32.00ms
step 890 | loss 0.0788 | step time 34.00ms
step 900 | loss 0.1051 | step time 34.00ms
step 910 | loss 0.0763 | step time 31.00ms
step 920 | loss 0.0917 | step time 35.00ms
step 930 | loss 0.0879 | step time 33.00ms
step 940 | loss 0.0808 | step time 33.00ms
step 950 | loss 0.0912 | step time 34.00ms
step 960 | loss 0.0764 | step time 66.00ms
step 970 | loss 0.0752 | step time 33.00ms
step 980 | loss 0.0888 | step time 41.00ms
step 990 | loss 0.0760 | step time 30.00ms
step 1000 | loss 0.0926 | step time 35.00ms
step 1000 train loss: 0.0857531875371933 test loss: 0.07519450783729553
test loss 0.07519450783729553 is the best so far, saving model to output/song_star\model.pt
--------------------------------------------------------------------------------
10 samples that are in train:
$
1155665 4433221
$
5544332 5544332
1155665 4433221
1155665 4433221
5544332 5544332
1155665 4433221
$
1155665 4433221
0 samples that are in test:
0 samples that are new:
--------------------------------------------------------------------------------
step 1010 | loss 0.0928 | step time 32.00ms
step 1020 | loss 0.0800 | step time 33.00ms
step 1030 | loss 0.0842 | step time 49.00ms
step 1040 | loss 0.0808 | step time 43.00ms
step 1050 | loss 0.0959 | step time 37.00ms
step 1060 | loss 0.0871 | step time 56.00ms
step 1070 | loss 0.0799 | step time 32.00ms
step 1080 | loss 0.0869 | step time 32.00ms
step 1090 | loss 0.0887 | step time 34.00ms
step 1100 | loss 0.1005 | step time 46.00ms
step 1110 | loss 0.0927 | step time 33.00ms
step 1120 | loss 0.0866 | step time 33.00ms
step 1130 | loss 0.0737 | step time 40.00ms
step 1140 | loss 0.0799 | step time 32.00ms
step 1150 | loss 0.0801 | step time 33.00ms
step 1160 | loss 0.0790 | step time 33.00ms
step 1170 | loss 0.0747 | step time 32.00ms
step 1180 | loss 0.0702 | step time 34.00ms
step 1190 | loss 0.0863 | step time 34.00ms
step 1200 | loss 0.0736 | step time 34.00ms
--------------------------------------------------------------------------------
10 samples that are in train:
$
$
1155665 4433221
$
$
$
1155665 4433221
1155665 4433221
1155665 4433221
5544332 5544332
0 samples that are in test:
0 samples that are new:
--------------------------------------------------------------------------------
step 1210 | loss 0.1007 | step time 40.00ms
step 1220 | loss 0.0958 | step time 33.00ms
step 1230 | loss 0.0930 | step time 62.00ms
step 1240 | loss 0.0980 | step time 38.00ms
step 1250 | loss 0.0913 | step time 35.02ms
step 1260 | loss 0.0941 | step time 33.00ms
step 1270 | loss 0.0864 | step time 214.01ms
step 1280 | loss 0.0886 | step time 35.00ms
step 1290 | loss 0.0922 | step time 34.00ms
step 1300 | loss 0.0930 | step time 33.00ms
step 1310 | loss 0.0823 | step time 36.00ms
step 1320 | loss 0.0858 | step time 51.00ms
step 1330 | loss 0.0814 | step time 33.00ms
step 1340 | loss 0.0925 | step time 33.99ms
step 1350 | loss 0.1038 | step time 33.00ms
step 1360 | loss 0.1030 | step time 34.00ms
step 1370 | loss 0.0788 | step time 33.00ms
step 1380 | loss 0.0805 | step time 39.01ms
step 1390 | loss 0.0963 | step time 37.00ms
step 1400 | loss 0.0910 | step time 35.00ms
--------------------------------------------------------------------------------
10 samples that are in train:
5544332 5544332
$
5544332 5544332
$
$
5544332 5544332
$
$
5544332 5544332
5544332 5544332
0 samples that are in test:
0 samples that are new:
--------------------------------------------------------------------------------
step 1410 | loss 0.0719 | step time 34.00ms
step 1420 | loss 0.0948 | step time 35.00ms
step 1430 | loss 0.0914 | step time 36.01ms
step 1440 | loss 0.0927 | step time 33.00ms
step 1450 | loss 0.0705 | step time 34.00ms
step 1460 | loss 0.0881 | step time 34.00ms
step 1470 | loss 0.0910 | step time 33.00ms
step 1480 | loss 0.0898 | step time 33.00ms
step 1490 | loss 0.0788 | step time 35.00ms
step 1500 | loss 0.0823 | step time 39.00ms
step 1500 train loss: 0.08480053395032883 test loss: 0.07294618338346481
test loss 0.07294618338346481 is the best so far, saving model to output/song_star\model.pt
step 1510 | loss 0.0893 | step time 32.00ms
step 1520 | loss 0.0798 | step time 34.00ms
step 1530 | loss 0.0965 | step time 33.00ms
step 1540 | loss 0.0941 | step time 64.00ms
step 1550 | loss 0.0759 | step time 36.01ms
step 1560 | loss 0.0588 | step time 32.00ms
step 1570 | loss 0.0822 | step time 34.02ms
step 1580 | loss 0.0852 | step time 33.00ms
step 1590 | loss 0.0942 | step time 33.00ms
step 1600 | loss 0.0874 | step time 34.00ms
--------------------------------------------------------------------------------
10 samples that are in train:
1155665 4433221
1155665 4433221
1155665 4433221
$
1155665 4433221
1155665 4433221
1155665 4433221
$
5544332 5544332
5544332 5544332
0 samples that are in test:
0 samples that are new:
--------------------------------------------------------------------------------
step 1610 | loss 0.0790 | step time 35.02ms
step 1620 | loss 0.0623 | step time 35.00ms
step 1630 | loss 0.1019 | step time 33.00ms
step 1640 | loss 0.0902 | step time 33.00ms
step 1650 | loss 0.0769 | step time 35.00ms
step 1660 | loss 0.0673 | step time 39.00ms
step 1670 | loss 0.0811 | step time 34.00ms
step 1680 | loss 0.0756 | step time 35.00ms
step 1690 | loss 0.0742 | step time 33.00ms
step 1700 | loss 0.0976 | step time 34.00ms
step 1710 | loss 0.0842 | step time 34.00ms
step 1720 | loss 0.0884 | step time 32.00ms
step 1730 | loss 0.1025 | step time 32.00ms
step 1740 | loss 0.0846 | step time 33.00ms
step 1750 | loss 0.0690 | step time 34.00ms
step 1760 | loss 0.0863 | step time 31.00ms
step 1770 | loss 0.0906 | step time 34.00ms
step 1780 | loss 0.0939 | step time 48.00ms
step 1790 | loss 0.0713 | step time 33.00ms
step 1800 | loss 0.0837 | step time 61.00ms
--------------------------------------------------------------------------------
10 samples that are in train:
1155665 4433221
1155665 4433221
$
5544332 5544332
1155665 4433221
1155665 4433221
$
1155665 4433221
5544332 5544332
1155665 4433221
0 samples that are in test:
0 samples that are new:
--------------------------------------------------------------------------------
step 1810 | loss 0.0840 | step time 42.00ms
step 1820 | loss 0.0908 | step time 35.00ms
step 1830 | loss 0.1031 | step time 49.00ms
step 1840 | loss 0.0692 | step time 53.00ms
step 1850 | loss 0.1011 | step time 33.00ms
step 1860 | loss 0.0893 | step time 41.00ms
step 1870 | loss 0.0893 | step time 62.00ms
step 1880 | loss 0.0795 | step time 34.00ms
step 1890 | loss 0.0794 | step time 37.00ms
step 1900 | loss 0.0841 | step time 54.00ms
step 1910 | loss 0.0824 | step time 37.00ms
step 1920 | loss 0.0666 | step time 61.00ms
step 1930 | loss 0.0875 | step time 57.00ms
step 1940 | loss 0.0865 | step time 34.00ms
step 1950 | loss 0.0779 | step time 51.00ms
step 1960 | loss 0.0721 | step time 49.00ms
step 1970 | loss 0.1079 | step time 45.01ms
step 1980 | loss 0.1123 | step time 70.00ms
step 1990 | loss 0.0924 | step time 39.01ms
step 2000 | loss 0.0977 | step time 33.00ms
step 2000 train loss: 0.0847245529294014 test loss: 0.07434774190187454
--------------------------------------------------------------------------------
10 samples that are in train:
$
$
5544332 5544332
1155665 4433221
$
1155665 4433221
$
1155665 4433221
$
$
0 samples that are in test:
0 samples that are new:
--------------------------------------------------------------------------------
step 2010 | loss 0.0770 | step time 32.00ms
step 2020 | loss 0.0877 | step time 34.00ms
step 2030 | loss 0.0837 | step time 47.00ms
step 2040 | loss 0.0819 | step time 51.00ms
step 2050 | loss 0.0798 | step time 53.00ms
step 2060 | loss 0.0885 | step time 31.00ms
step 2070 | loss 0.0854 | step time 63.01ms
step 2080 | loss 0.0968 | step time 45.00ms
step 2090 | loss 0.0844 | step time 49.00ms
step 2100 | loss 0.0964 | step time 45.00ms
step 2110 | loss 0.0758 | step time 40.00ms
step 2120 | loss 0.0872 | step time 94.01ms
step 2130 | loss 0.0907 | step time 53.00ms
step 2140 | loss 0.1142 | step time 53.01ms
step 2150 | loss 0.0953 | step time 45.00ms
step 2160 | loss 0.1223 | step time 44.00ms
step 2170 | loss 0.0986 | step time 82.00ms
step 2180 | loss 0.0836 | step time 95.00ms
step 2190 | loss 0.0746 | step time 50.00ms
step 2200 | loss 0.0948 | step time 38.00ms
--------------------------------------------------------------------------------
10 samples that are in train:
1155665 4433221
5544332 5544332
5544332 5544332
$
5544332 5544332
1155665 4433221
$
1155665 4433221
1155665 4433221
$
0 samples that are in test:
0 samples that are new:
--------------------------------------------------------------------------------
step 2210 | loss 0.0785 | step time 45.00ms
step 2220 | loss 0.0880 | step time 68.00ms
step 2230 | loss 0.1060 | step time 54.00ms
step 2240 | loss 0.0642 | step time 59.00ms
step 2250 | loss 0.0915 | step time 49.00ms
step 2260 | loss 0.0843 | step time 56.00ms
step 2270 | loss 0.0872 | step time 52.00ms
step 2280 | loss 0.0872 | step time 102.01ms
step 2290 | loss 0.1017 | step time 44.00ms
step 2300 | loss 0.0884 | step time 174.01ms
step 2310 | loss 0.0835 | step time 60.00ms
step 2320 | loss 0.0979 | step time 57.00ms
step 2330 | loss 0.0816 | step time 76.00ms
step 2340 | loss 0.0798 | step time 49.00ms
step 2350 | loss 0.0818 | step time 55.00ms
step 2360 | loss 0.0603 | step time 75.00ms
step 2370 | loss 0.0760 | step time 52.00ms
step 2380 | loss 0.0887 | step time 48.00ms
step 2390 | loss 0.0735 | step time 43.00ms
step 2400 | loss 0.0934 | step time 49.00ms
--------------------------------------------------------------------------------
10 samples that are in train:
1155665 4433221
$
1155665 4433221
5544332 5544332
1155665 4433221
$
$
5544332 5544332
5544332 5544332
1155665 4433221
0 samples that are in test:
0 samples that are new:
--------------------------------------------------------------------------------
step 2410 | loss 0.0716 | step time 66.01ms
step 2420 | loss 0.0817 | step time 39.00ms
step 2430 | loss 0.0981 | step time 34.00ms
step 2440 | loss 0.0754 | step time 85.00ms
step 2450 | loss 0.0618 | step time 33.00ms
step 2460 | loss 0.0717 | step time 38.00ms
step 2470 | loss 0.1001 | step time 38.00ms
step 2480 | loss 0.0755 | step time 103.01ms
step 2490 | loss 0.0906 | step time 34.00ms
step 2500 | loss 0.0788 | step time 35.00ms
step 2500 train loss: 0.08496858179569244 test loss: 0.07424865663051605
step 2510 | loss 0.1235 | step time 34.00ms
step 2520 | loss 0.0850 | step time 53.00ms
step 2530 | loss 0.0878 | step time 45.00ms
step 2540 | loss 0.0705 | step time 33.00ms
step 2550 | loss 0.0925 | step time 32.00ms
step 2560 | loss 0.1008 | step time 32.00ms
step 2570 | loss 0.0880 | step time 34.00ms
step 2580 | loss 0.1090 | step time 33.00ms
step 2590 | loss 0.0863 | step time 34.00ms
step 2600 | loss 0.0674 | step time 36.00ms
--------------------------------------------------------------------------------
10 samples that are in train:
1155665 4433221
5544332 5544332
1155665 4433221
$
1155665 4433221
1155665 4433221
$
$
5544332 5544332
1155665 4433221
0 samples that are in test:
0 samples that are new:
--------------------------------------------------------------------------------
step 2610 | loss 0.1034 | step time 35.00ms
step 2620 | loss 0.0885 | step time 47.00ms
step 2630 | loss 0.0854 | step time 31.00ms
step 2640 | loss 0.1056 | step time 33.99ms
step 2650 | loss 0.0867 | step time 33.00ms
step 2660 | loss 0.0827 | step time 35.00ms
step 2670 | loss 0.0677 | step time 33.00ms
step 2680 | loss 0.0843 | step time 33.00ms
step 2690 | loss 0.0972 | step time 33.00ms
step 2700 | loss 0.0941 | step time 32.99ms
step 2710 | loss 0.0814 | step time 35.00ms
step 2720 | loss 0.0827 | step time 68.00ms
step 2730 | loss 0.0771 | step time 33.00ms
step 2740 | loss 0.0840 | step time 34.00ms
step 2750 | loss 0.0839 | step time 34.00ms
step 2760 | loss 0.0723 | step time 33.00ms
step 2770 | loss 0.0839 | step time 32.00ms
step 2780 | loss 0.0867 | step time 36.00ms
step 2790 | loss 0.0709 | step time 34.00ms
step 2800 | loss 0.0849 | step time 37.00ms
--------------------------------------------------------------------------------
10 samples that are in train:
1155665 4433221
$
1155665 4433221
5544332 5544332
1155665 4433221
$
$
$
$
1155665 4433221
0 samples that are in test:
0 samples that are new:
--------------------------------------------------------------------------------
step 2810 | loss 0.0845 | step time 34.00ms
step 2820 | loss 0.0684 | step time 35.00ms
step 2830 | loss 0.0907 | step time 40.00ms
step 2840 | loss 0.0925 | step time 34.00ms
step 2850 | loss 0.0850 | step time 40.01ms
step 2860 | loss 0.0861 | step time 35.00ms
step 2870 | loss 0.0708 | step time 34.00ms
step 2880 | loss 0.0695 | step time 35.00ms
step 2890 | loss 0.1012 | step time 32.00ms
step 2900 | loss 0.0759 | step time 46.00ms
step 2910 | loss 0.0965 | step time 36.00ms
step 2920 | loss 0.0813 | step time 34.00ms
step 2930 | loss 0.0883 | step time 34.00ms
step 2940 | loss 0.0882 | step time 34.00ms
step 2950 | loss 0.0881 | step time 32.00ms
step 2960 | loss 0.0690 | step time 33.00ms
step 2970 | loss 0.1002 | step time 33.00ms
step 2980 | loss 0.0767 | step time 45.00ms
step 2990 | loss 0.0968 | step time 31.00ms
step 3000 | loss 0.1164 | step time 44.00ms
step 3000 train loss: 0.08447916805744171 test loss: 0.07339181751012802
--------------------------------------------------------------------------------
10 samples that are in train:
1155665 4433221
1155665 4433221
1155665 4433221
1155665 4433221
5544332 5544332
1155665 4433221
1155665 4433221
5544332 5544332
$
$
0 samples that are in test:
0 samples that are new:
--------------------------------------------------------------------------------
step 3010 | loss 0.0887 | step time 35.00ms
step 3020 | loss 0.0891 | step time 35.00ms
step 3030 | loss 0.0853 | step time 72.00ms
step 3040 | loss 0.0942 | step time 38.00ms
step 3050 | loss 0.0865 | step time 34.00ms
step 3060 | loss 0.0892 | step time 34.00ms
step 3070 | loss 0.0860 | step time 36.00ms
step 3080 | loss 0.0884 | step time 32.01ms
step 3090 | loss 0.0746 | step time 33.00ms
step 3100 | loss 0.0713 | step time 33.00ms
step 3110 | loss 0.0988 | step time 32.00ms
step 3120 | loss 0.0804 | step time 46.00ms
step 3130 | loss 0.0703 | step time 34.00ms
step 3140 | loss 0.0909 | step time 33.00ms
step 3150 | loss 0.0858 | step time 31.00ms
step 3160 | loss 0.1123 | step time 33.99ms
step 3170 | loss 0.0905 | step time 43.00ms
step 3180 | loss 0.0767 | step time 34.00ms
step 3190 | loss 0.0842 | step time 34.00ms
step 3200 | loss 0.1024 | step time 34.00ms
--------------------------------------------------------------------------------
10 samples that are in train:
5544332 5544332
$
$
1155665 4433221
$
1155665 4433221
1155665 4433221
5544332 5544332
$
5544332 5544332
0 samples that are in test:
0 samples that are new:
--------------------------------------------------------------------------------
step 3210 | loss 0.0937 | step time 32.00ms
step 3220 | loss 0.0858 | step time 34.00ms
step 3230 | loss 0.0784 | step time 34.00ms
step 3240 | loss 0.1014 | step time 37.00ms
step 3250 | loss 0.0860 | step time 34.99ms
step 3260 | loss 0.0956 | step time 33.00ms
step 3270 | loss 0.0690 | step time 32.99ms
step 3280 | loss 0.0788 | step time 35.00ms
step 3290 | loss 0.0678 | step time 34.00ms
step 3300 | loss 0.0705 | step time 33.00ms
step 3310 | loss 0.0746 | step time 35.00ms
step 3320 | loss 0.0832 | step time 34.00ms
step 3330 | loss 0.0770 | step time 39.00ms
step 3340 | loss 0.0888 | step time 34.00ms
step 3350 | loss 0.0686 | step time 33.00ms
step 3360 | loss 0.0769 | step time 33.00ms
step 3370 | loss 0.0864 | step time 33.00ms
step 3380 | loss 0.0751 | step time 37.00ms
step 3390 | loss 0.0993 | step time 32.01ms
step 3400 | loss 0.0710 | step time 33.00ms
--------------------------------------------------------------------------------
10 samples that are in train:
$
$
5544332 5544332
1155665 4433221
$
1155665 4433221
1155665 4433221
5544332 5544332
$
$
0 samples that are in test:
0 samples that are new:
--------------------------------------------------------------------------------
step 3410 | loss 0.0733 | step time 50.00ms
step 3420 | loss 0.0859 | step time 35.00ms
step 3430 | loss 0.0837 | step time 34.00ms
step 3440 | loss 0.0745 | step time 34.00ms
step 3450 | loss 0.0929 | step time 35.00ms
step 3460 | loss 0.0690 | step time 32.00ms
step 3470 | loss 0.0789 | step time 31.00ms
step 3480 | loss 0.1055 | step time 36.00ms
step 3490 | loss 0.1055 | step time 34.00ms
step 3500 | loss 0.0965 | step time 88.01ms
step 3500 train loss: 0.08492360264062881 test loss: 0.07304983586072922
step 3510 | loss 0.0924 | step time 33.00ms
step 3520 | loss 0.0915 | step time 34.00ms
step 3530 | loss 0.0880 | step time 68.00ms
step 3540 | loss 0.0893 | step time 35.00ms
step 3550 | loss 0.0879 | step time 31.00ms
step 3560 | loss 0.0926 | step time 33.00ms
step 3570 | loss 0.0753 | step time 34.00ms
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
  File "D:\ccc\py2gpt\04c-makemore\makemore.py", line 273, in forward
    xh = torch.cat([xt, hprev], dim=1)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
KeyboardInterrupt