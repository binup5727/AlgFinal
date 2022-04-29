from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt 
import time


def cutBruteForce(n, p):
    #creates all possible cuts
    print(n)
    maxp = 0
    timeA = datetime.now()
    steps = 0
    for i in range(2**(n-1)):
        #print(2**(n-1))
        binaryArr = np.array(list(np.binary_repr(i).zfill(n-1))).astype(np.int8)
        
        length2 = 0
        
        locp = 0
        for j in range(len(binaryArr)):
            steps += 1
            print(steps, " of {:e}".format(2**(n-1)*n))
            
            length2 += 1
            #print(locp)
            if binaryArr[j] == 1:
                locp = locp + p[length2 - 1]
                length2 = 0
                #print(locp)
            if j == n-2:
                
                length2 += 1
                locp = locp + p[length2 - 1]

                
                length2 = 0
        if locp > maxp:
            maxp = locp
            maxcuts = binaryArr
        
    time = datetime.now() - timeA
    return maxp, steps, time
            
        

        
def cutBottomUp(n, p):
    r = [0 for i in range(n + 1)]
    steps = 0
    timeA = datetime.now()
    #print(r)
    maxp = 0
    for j in range(1, n + 1):
        maxp = 0
        for i in range(j):
            steps += 1
            maxp = max(maxp, p[i] + r[j - i - 1])
        r[j] = maxp

    time = datetime.now() - timeA
    return r[n], steps, time

    







n = [i*(10**2) for i in range(1,11)]
print(n)
constant = 2

p = [1]
for i in range(1, n[-1]):
        p.append(p[i - 1] + constant) 

Alg1_Steps = np.zeros(10)
Alg2_Steps = np.zeros(10)
Alg1_Time = np.zeros(10)
Alg2_Time = np.zeros(10) 
for j in range(len(n)):
    #make price values increase by constant
    



    profit, Alg1_Steps[j], Alg1_Time[j] = cutBruteForce(n[j], p)

    print(profit, Alg1_Steps[j], Alg1_Time[j])


    profit2, Alg2_Steps[j], Alg2_Time[j] = cutBottomUp(n, p)

    print(profit2, Alg2_Steps[j], Alg2_Time[j])
















