# gdGate.py

```
f(i) = o

learn f:
give 

i0 => o0      f(i0) => o0'
i1 => o1      f(i1) => o1'
i2 => o2      f(i2) => o2'
.....
in => on      f(in) => on'
delta = [o0-o0', o1-o1', ...., on-on']

loss = norm(delta)
```

調整 w 後得到 f 使 delta 愈小愈好！