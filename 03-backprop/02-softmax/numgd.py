def df(f, p, k, h=0.001):
    p1 = p.copy()
    p1[k] = p[k]+h
    return (f(p1) - f(p)) / h

# 函數 f 在點 p 上的梯度
def grad(f, p, h=0.001):
    gp = p.copy()
    for k in range(len(p)):
        gp[k] = df(f, p, k, h)
    return gp


