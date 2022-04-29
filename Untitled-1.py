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
    return maxp, steps, time.microseconds
            
        

        
def cutBottomUp(n, p):
    r = [0] * (n + 1)
    
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
    return r[n], steps, time.microseconds

    
#print(cutBottomUp(4, [1,4,6,5,6,7,99]))






n = [i*(10**1) for i in range(1,11)]
print(n)
n = np.array(n)
constant = 2

p = [1]
for i in range(1, n[-1]):
        p.append(p[i - 1] + constant) 

Alg1_Steps = np.zeros(10)
Alg2_Steps = np.zeros(10)

Alg1_Time = np.zeros(10)
Alg2_Time = np.zeros(10) 

Alg1_StepRatio = np.zeros(10)
Alg2_StepRatio = np.zeros(10)

Alg1_TimeRatio = np.zeros(10)
Alg2_TimeRatio = np.zeros(10)

TheoreticalAlg1 = np.zeros(10)
TheoreticalAlg2 = np.zeros(10)

for j in range(len(n)):
    #make price values increase by constant
    
    TheoreticalAlg1[j] = n[j] * (2**(n[j]-1))
    TheoreticalAlg2[j] = n[j]**2


    profit, Alg1_Steps[j], Alg1_Time[j] = cutBruteForce(n[j], p)
    Alg1_StepRatio[j] = Alg1_Steps[j]/TheoreticalAlg1[j]
    Alg1_TimeRatio[j] = Alg1_Time[j]/TheoreticalAlg1[j]

    print(profit, Alg1_Steps[j], Alg1_Time[j])



    profit2, Alg2_Steps[j], Alg2_Time[j] = cutBottomUp(n[j], p)
    Alg2_StepRatio[j] = Alg2_Steps[j]/TheoreticalAlg2[j]
    Alg2_TimeRatio[j] = Alg2_Time[j]/TheoreticalAlg2[j]


    
    print(profit2, Alg2_Steps[j], Alg2_Time[j])

Alg1Cstep = max(Alg1_StepRatio)
Alg1Ctime = max(Alg1_TimeRatio)

Alg2Cstep = max(Alg2_StepRatio)
Alg2Ctime = max(Alg2_TimeRatio)

PredictedRT_Alg1_Step = Alg1Cstep * TheoreticalAlg1
PredictedRT_Alg1_time = Alg1Ctime * TheoreticalAlg1

PredictedRT_Alg2_Step = Alg2Cstep * TheoreticalAlg2
PredictedRT_Alg2_time = Alg2Ctime * TheoreticalAlg2

plt.plot(n, Alg1_Steps, label="EmpiricalRT Brute Force")
plt.plot(n, PredictedRT_Alg1_Step, label="PredictedRT Brute Force")
plt.xlabel("n values")
plt.ylabel("RunTime in Steps")
plt.legend()
plt.show()

plt.plot(n, Alg1_Time, label="EmpiricalRT Brute Force")
plt.plot(n, PredictedRT_Alg1_time, label="PredictedRT Brute Force")
plt.xlabel("n values")
plt.ylabel("RunTime in Time(ms)")
plt.legend()
plt.show()



plt.plot(n, Alg2_Steps, label="EmpiricalRT DP Bottom Up")
plt.plot(n, PredictedRT_Alg2_Step, label="PredictedRT DP Bottom Up")
plt.xlabel("n values")
plt.ylabel("RunTime in Steps")
plt.legend()
plt.show()

plt.plot(n, Alg2_Time, label="EmpiricalRT DP Bottom Up")
plt.plot(n, PredictedRT_Alg2_time, label="PredictedRT DP Bottom Up")
plt.xlabel("n values")
plt.ylabel("RunTime in Time(ms)")
plt.legend()
plt.show()



plt.plot(n, Alg1_Steps, label="EpriricalRT Brute Force")
plt.plot(n, Alg2_Steps, label="EmpiricalRT DP Bottom-Up")
plt.xlabel("n values")
plt.ylabel("RunTime in Steps")
plt.legend()
plt.show()

plt.plot(n, Alg1_Time, label="EpriricalRT Brute Force")
plt.plot(n, Alg2_Time, label="EmpiricalRT DP Bottom-Up")
plt.xlabel("n values")
plt.ylabel("RunTime in Time(ms)")
plt.legend()
plt.show()









