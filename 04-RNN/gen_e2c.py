from english import S, mt, e2c

for _ in range(1000):
    s = S()
    s1 = mt(e2c, s)
    print('en: '+' '.join(s)+' tw: '+' '.join(s1))

