from english import S

for _ in range(1000):
    s = S()
    q = ' '.join(s)
    a = filter(lambda w: w in ['dog', 'cat', 'pig', 'chase', 'bite', 'love'], s)
    print('Q: '+q+' A: '+' '.join(a))


