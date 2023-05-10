def square(x):
    return x*x

def power3(x):
    return x*x*x

def triple(x):
    return 3*x

dx = 0.001

def diff(f, x):
    df = f(x+dx)-f(x)
    return df/dx

print('diff(square,2)=', diff(square, 2))
print('diff(power3,1)=', diff(power3, 1))
print('diff(power3,2)=', diff(power3, 2))
print('diff(triple,1)=', diff(triple, 1))
print('diff(triple,2)=', diff(triple, 2))
