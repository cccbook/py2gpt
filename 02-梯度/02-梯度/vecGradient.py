step = 0.01

# 我們想找函數 f 的最低點
def f(p):
    [x,y] = p
    return x * x + y * y

# df(f, p, k) 為函數 f 對變數 k 的偏微分: df / dp[k]
# 例如在上述 f 範例中 k=0, df/dx, k=1, df/dy
def df(f, p, k):
    p1 = p.copy()
    p1[k] += step
    return (f(p1) - f(p)) / step

# 函數 f 在點 p 上的梯度
def grad(f, p):
    gp = p.copy()
    for k in range(len(p)):
        gp[k] = df(f, p, k)
    return gp

[x,y] = [1,3]
print('x=', x, 'y=', y)
print('df(f(x,y), 0) = ', df(f, [x, y], 0))
print('df(f(x,y), 1) = ', df(f, [x, y], 1))
print('grad(f)=', grad(f, [x,y]))
