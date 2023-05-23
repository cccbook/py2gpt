# Softmax + CrossEntropy

* 參考 -- https://mattpetersen.github.io/softmax-with-cross-entropy

## test1.py

驗證 softmax + cross_entropy 的反傳遞公式，就是 s - y 。

其中 s = softmax(x)

```py
$ python test1.py
x = [0.3 0.5 0.2]
y = [0. 1. 0.]
s = softmax(x) = [0.31987306 0.39069383 0.28943311]
jacobian_softmax(s)=
 [[ 0.21755428 -0.12497243 -0.09258185]
 [-0.12497243  0.23805216 -0.11307973]
 [-0.09258185 -0.11307973  0.20566159]]
cross_entropy(y, s)= [0.93983106]
    gradient_cross_entropy(y, s)= [-0.         -2.55954897 -0.        ]
num_gradient_cross_entropy(y, s)= [ 0.         -2.55627891  0.        ]
    error_softmax_input(y, s)= [ 0.31987306 -0.60930617  0.28943311]
num_error_softmax_input(y, x)= [ 0.31998185 -0.60918713  0.28953596]
```