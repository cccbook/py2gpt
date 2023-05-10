import gd2 as gd

def f(p):
    [x] = p
    return abs(x*x-4) # 能量函數為 |x^2-4|

p = [1.0]
gd.gradientDescendent(f, p)
