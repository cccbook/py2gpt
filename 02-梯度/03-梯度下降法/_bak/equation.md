# gdEquation.py

```
>>> import numpy as np
>>> A = np.array([[3.0, 2],[1, 1]])
>>> B = np.array([5.0, 2])
>>> x = np.array([1.0, 1])
>>> A.dot(X)
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
NameError: name 'X' is not defined
>>> A.dot(x) 
array([5., 2.])
>>> xt = x.transpose()
>>> A.dot(xt) 
array([5., 2.])
```