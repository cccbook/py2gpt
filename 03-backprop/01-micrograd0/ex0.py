from micrograd.engine import Value

x = Value(-2.0)
y = Value(5.0)
z = Value(-4.0)

q = x+y
f = q*z

print('====== forward ======')
print('x=', x)
print('y=', y)
print('z=', z)
print('q=', q)
print('f=', f)

print('====== backward ======')
f.backward()
print('x=', x)
print('y=', y)
print('z=', z)
print('q=', q)
print('f=', f)

'''
$ python ex0.py
====== forward ======
x= Value(data=-2.0, grad=0)
y= Value(data=5.0, grad=0)
z= Value(data=-4.0, grad=0)
q= Value(data=3.0, grad=0)
f= Value(data=-12.0, grad=0)
====== backward ======
x= Value(data=-2.0, grad=-4.0)
y= Value(data=5.0, grad=-4.0)
z= Value(data=-4.0, grad=3.0)
q= Value(data=3.0, grad=-4.0)
f= Value(data=-12.0, grad=1)
'''

