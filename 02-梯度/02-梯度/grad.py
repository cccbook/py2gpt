h = 0.01

# df(f, p, k) 為函數 f 對變數 k 的偏微分: df / dp[k]
# 例如在 p 是二維點的情況下，函數 f(p) = f(x,y) 的情況
#     k=0 時偏微分是 df/dx, k=1 時偏微分是 df/dy
def df(f, p, k):
    p1 = p.copy()
    p1[k] += h
    return (f(p1) - f(p)) / h

# 函數 f 在點 p 上的梯度
def grad(f, p):
    gp = p.copy()
    for k in range(len(p)):
        gp[k] = df(f, p, k)
    return gp

 # f(x,y)=x*x+y*y
def f(p):
    [x,y] = p
    return x * x + y * y

p = [x,y] = [1,3]
print('p=', p) # p = [x,y] = [1, 3], 我們想算這一點的梯度
print('df(f, 0) = ', df(f, p, 0)) # x 方向的偏微分 df/dx = 2
print('df(f, 1) = ', df(f, p, 1)) # y 方向的偏微分 df/dy = 6
print('grad(f)=', grad(f, p)) # 梯度 = (df/dx, df/dy) = (2,6)

'''
執行結果

$ python grad.py
p= [1, 3]
df(f, 0) =  2.009999999999934
df(f, 1) =  6.009999999999849
grad(f)= [2.009999999999934, 6.009999999999849]
'''
