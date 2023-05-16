import random as r

'''
E = N | E [+/-*] E
N = 0-9
'''

def E():
	gen = r.choice(["N", "N", "EE"])
	# print('gen=', gen)
	if gen == "N":
		return N()
	else:
		return "( " + E() + ' ' + r.choice(["+", "-", "*", "/"]) + ' ' + E() + " )"

def N():
	return r.choice(["1", "2", "3", "4", "5", "6", "7", "8", "9"])

for _ in range(1, 3000):
    e = E()
    if len(e) > 1:
        print(e)
