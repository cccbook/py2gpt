h = 0.001

def diff(f, x):
    df = f(x+h)-f(x)
    return df/h

def triple(x):
    return 3*x

def square(x):
    return x**2

def power3(x):
    return x**3

print('diff(triple,2)=', diff(triple, 2)) # 3x 的微分為 3，不管 x 多少斜率都是 3
print('diff(square,2)=', diff(square, 2)) # x**2 的微分為 2x, x=2 時斜率為 2*2 = 4
print('diff(power3,2)=', diff(power3, 2)) # x**3 的微分為 3x**2, x=2 時斜率為 3*2*2 = 12

'''
執行結果 

$ python diff.py
diff(triple,2)= 3.0000000000001137
diff(square,2)= 4.000999999999699
diff(power3,2)= 12.006000999997823
'''
