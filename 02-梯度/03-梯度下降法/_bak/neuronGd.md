AND gate
x   y | o = and(x,y)
------|---
0   0 | 0     x0 y0 o0
0   1 | 0     x1 y1 o1
1   0 | 0     x2 y2 o2
1   1 | 1     x3 y3 o3

sig(w1*x + w2*y + b) = o
sig(w1*0 + w2*0 + b) = o0   => 0
sig(w1*0 + w2*1 + b) = o1   => 0
sig(w1*1 + w2*0 + b) = o2   => 0
sig(w1*1 + w2*1 + b) = o3   => 1


f(x,y,w) = (o0-0)^2 + (o1-0)^2 + (o2-0)^2 + (o3-1)^2 

sig 是 sigmoid 函數

* https://zh.wikipedia.org/wiki/S%E5%87%BD%E6%95%B0