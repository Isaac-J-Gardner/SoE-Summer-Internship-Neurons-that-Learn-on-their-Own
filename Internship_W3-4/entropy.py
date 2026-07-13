import math

while True:
    k = input("enter value for k")
    if int(k):
        k = int(k)
        p = 1/k
        h = 0
        for i in range(k):
            h += p * math.log(p)
        h = -1 * h
        print(h, math.exp(h))
    else:
        break
