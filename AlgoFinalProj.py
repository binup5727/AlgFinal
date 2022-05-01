
import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd
import math



def cutBruteForce(n, p):

    maxp = 0
    
    steps = 0
    num = int(math.pow(2, (n-1)))
    for i in range(num):
        
        binaryArr = np.array(list(np.binary_repr(i).zfill(n-1))).astype(np.int8)
        
        length2 = 0
        
        locp = 0
        for j in range(len(binaryArr)):
            steps += 1
            print(n)
            print(steps, " of {:e}".format(num * len(binaryArr)))
            
            length2 += 1
           
            if binaryArr[j] == 1:
                locp = locp + p[length2 - 1]
                length2 = 0
            
            if j == n-2:
                
                length2 += 1
                locp = locp + p[length2 - 1]

                
                length2 = 0
        if locp > maxp:
            maxp = locp
            maxcuts = binaryArr
        
    
    return maxp, steps
            
        

        
def cutBottomUp(n, p):
    r = [0] * (n + 1)
    
    steps = 0
    

    maxp = 0
    for j in range(1, n + 1):
        maxp = 0
        for i in range(j):
            print(n)
            print(steps, " of {:e}".format(n**2))
            steps += 1
            maxp = max(maxp, p[i] + r[j - i - 1])
        r[j] = maxp

    
    maxp = r[n]

    return r[n], steps






size = 6
m = 3
multiplier = 1
n = [i * (5**multiplier) for i in range(1, size + 1)]
print(n)
n = np.array(n)
constant = 2

p = [1]
for i in range(1, n[-1]):
        p.append(p[i - 1] + constant) 

Alg1_Steps = np.zeros(size)
Alg2_Steps = np.zeros(size)


Alg1_StepRatio = np.zeros(size)
Alg2_StepRatio = np.zeros(size)


TheoreticalAlg1 = np.zeros(size)
TheoreticalAlg2 = np.zeros(size)

profit = np.zeros(size)
profit2 = np.zeros(size)

for j in range(len(n)):
    
    TheoreticalAlg1[j] += n[j] * (math.pow(2, (n[j]) - 1))
    TheoreticalAlg2[j] += n[j]**2


for k in range(m):
    for j in range(len(n)):
        
        tup1 = cutBruteForce(n[j], p)
   
        profit[j] += tup1[0]
        Alg1_Steps[j] += tup1[1]
     
        tup2 = cutBottomUp(n[j], p)
        profit2[j] += tup2[0]
        Alg2_Steps[j] += tup2[1]
        
        



Alg1_Steps = Alg1_Steps/m


Alg2_Steps = Alg2_Steps/m


Alg1_StepRatio = Alg1_Steps/TheoreticalAlg1




Alg2_StepRatio = Alg2_Steps/TheoreticalAlg2




Alg1Cstep = max(Alg1_StepRatio[1:])


Alg2Cstep = max(Alg2_StepRatio[1:])


PredictedRT_Alg1_Step = Alg1Cstep * TheoreticalAlg1


PredictedRT_Alg2_Step = Alg2Cstep * TheoreticalAlg2

n1values = pd.DataFrame({'n': TheoreticalAlg1})
n2values = pd.DataFrame({"n": TheoreticalAlg2})

Alg1data = pd.DataFrame({'n':n, 'Empirical RT steps': Alg1_Steps, 'Ratio = (Emperical RT)/(Theoretical Complexity)': Alg1_StepRatio, 'predicted RT': PredictedRT_Alg1_Step})

Alg2data = pd.DataFrame({'n':n, 'Empirical RT steps': Alg2_Steps, 'Ratio = (Emperical RT)/(Theoretical Complexity)': Alg2_StepRatio, 'predicted RT': PredictedRT_Alg2_Step})
print(n1values)
print(n2values)
print(Alg1data)
print(Alg2data)
print(Alg1Cstep)
print(Alg2Cstep)

plt.plot(n, Alg1_Steps, label="EmpiricalRT Brute Force")
plt.plot(n, PredictedRT_Alg1_Step, label="PredictedRT Brute Force")
plt.xlabel("n values")
plt.ylabel("RunTime in Steps")
plt.legend()
plt.show()




plt.plot(n, Alg2_Steps, label="EmpiricalRT DP Bottom Up")
plt.plot(n, PredictedRT_Alg2_Step, label="PredictedRT DP Bottom Up")
plt.xlabel("n values")
plt.ylabel("RunTime in Steps")
plt.legend()
plt.show()





plt.plot(n, Alg1_Steps, label="EpriricalRT Brute Force")
plt.plot(n, Alg2_Steps, label="EmpiricalRT DP Bottom-Up")
plt.yscale("log")
plt.xlabel("n values")
plt.ylabel("RunTime in Steps")
plt.legend()
plt.show()










