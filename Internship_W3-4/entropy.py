#im reading up on information theory for some tests looking at cross entropy, and as a paper looked into redundancy and the optimum amount of it.
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
