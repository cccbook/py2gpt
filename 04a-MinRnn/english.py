import random as r

'''
S = NP VP
NP = DET N
VP = V NP
N = dog | cat
V = chase | eat
DET = a | the
'''

def S():
    s = []
    NP(s)
    if r.choice(['', 'vp'])=='vp':
        VP(s)
    return s

def NP(s):
    DET(s)
    if r.choice(['', 'adj'])=='adj':
        ADJ(s)
    N(s)

def VP(s):
    V(s)
    # if r.choice(['', 'np'])=='np':
    NP(s)

def N(s):
    s.append(r.choice(['dog', 'cat', 'pig']))

def V(s):
    s.append(r.choice(['chase', 'bite', 'love']))

def DET(s):
    s.append(r.choice(['a', 'the', 'every']))

def ADJ(s):
    s.append(r.choice(['black', 'white', 'little']))

e2c = { 'dog':'狗', 'cat':'貓', 'pig':'豬', 'a': '一隻', 'the': '這隻', 'every':'每隻', 'chase':'追', 'bite':'咬', 'love':'愛', 'white':'白', 'black':'黑', 'little':'小' }

def mt(s2t, elist): 
    clist = []
    for e in elist:
        c = s2t[e]
        clist.append(c)
    return clist
