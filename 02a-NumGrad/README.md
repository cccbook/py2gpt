# 專案 -- NumGrad

> 梯度下降法實作，用數值的方法計算梯度

梯度公式：

$$
\nabla_{x} f(x) = \left[ \frac{\partial }{\partial x_1} f(x), \frac{\partial }{\partial x_2} f(x),\cdots,\frac{\partial }{\partial x_n} f(x) \right]^T=\frac{\partial }{\partial{x}} f(x)
$$

偏微分公式：

$$
\frac{\partial }{\partial x_1} f(x) = \lim_{h \to 0} \frac{f(x_1, ..., x_i+h, ...., x_n)-f(x_1, ..., x_i, ...., x_n)}{h}
$$

於是我們可以用以下函數計算偏微分

```py
# 函數 f 對變數 k 的偏微分: df / dk
def df(f, p, k, h=0.001):
    p1 = p.copy()
    p1[k] = p[k]+h
    return (f(p1) - f(p)) / h
```

然後再將這些偏微分合成一個梯度向量

```py
# 函數 f 在點 p 上的梯度
def grad(f, p, h=0.001):
    gp = p.copy()
    for k in range(len(p)):
        gp[k] = df(f, p, k, h)
    return gp
```

最後透過一直往逆梯度方向前進的走法，找到函數的區域最低點 (谷底)。

```py

# 使用梯度下降法尋找函數最低點
def gradientDescendent(f, p0, step=0.009, max_loops=100000, dump_period=1000):
    p = np.array(p0)

    for i in range(max_loops):
        fp = f(p)       # 目前的函數值
        gp = grad(f, p) # 計算梯度 
        glen = norm(gp) # 梯度的長度 (步伐大小)
        if i%dump_period == 0: # 每執行 dump_period 次才會印出一次結果
            print('{:05d}:f(p)={:.3f} p={:s} gp={:s} glen={:.5f}'.format(i, fp, str(p), str(gp), glen))
        if glen < 0.00001: # or fp0 < fp:  # 如果步伐已經很小了，或者 f(p) 變大了，那麼就停止吧！
            break
        p +=  np.multiply(gp, -1*step) # 朝逆梯度方向的一小步

    print('{:05d}:f(p)={:.3f} p={:s} gp={:s} glen={:.5f}'.format(i, fp, str(p), str(gp), glen))
    return p # 傳回最低點！
```
