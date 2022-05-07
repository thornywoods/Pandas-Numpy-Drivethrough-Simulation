

#Scenario:
 # A counter with a random service time and customers who renege. Based on the
  #program bank08.py from TheBank tutorial of SimPy 2. (KGM)


import random

import simpy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


RANDOM_SEED = 747
arrivRate = 1.0 #car shows up every arrivRate minutes
ORDER_WINDOWS = 1
bailedOut = 0
REP = 100 #number of iterations on any given arrival rate
STEPS = 26 #how many tenths to examine of the arrival rate, starting at 1.0


def source(env, interval, orderWindow):
    """Source generates customers randomly"""
    i = 0;
    while True:
        c = customer(env, 'Customer%02d' % i, orderWindow)
        i += 1
        env.process(c)
        t = random.expovariate(1.0 / interval)
        yield env.timeout(t)


def customer(env, name, orderWindow):
    """Customer arrives, is served and leaves."""
   # print('%7.4f %s: arrived' % (arrive, name))
    #print(f'orderLine queue: {len(orderLine.queue)}')
    #print(f'orderWindow count: {orderWindow.count}')
    #check if the line before the order window is more than 7
    if(len(orderLine.queue) < 1):
        
        olReq = orderLine.request()
        yield olReq
        

        orReq = orderWindow.request()
        yield orReq
            
        yield orderLine.release(olReq)
        
        
        #print('%7.4f %s: Waited %6.3f' % (env.now, name, wait))
        tib = random.weibullvariate(3, 1.5)
        
        yield env.timeout(tib)
        
        
        cookingTime = random.weibullvariate(6, 2)
        wait = env.now + cookingTime
        #print('%7.4f %s: Finished at %7.4f' % (env.now, name, wait))
        plReq = payLine.request()
        yield plReq
        
        
    
        yield orderWindow.release(orReq)
        
        pyReq = paymentStation.request()
        yield pyReq
        #at the payment station
        
        yield payLine.release(plReq)
        
        tib = random.weibullvariate(2, 1.5)
        yield env.timeout(tib)
        
        pklReq = pickupLine.request()
        yield pklReq
        
        yield paymentStation.release(pyReq)
        pwReq = pickupWindow.request()
        yield pwReq
        
        yield pickupLine.release(pklReq)
        
        #print(f'Time until food done {wait} current time {env.now}')
        while(env.now < wait):
            yield env.timeout(.1)

        yield env.timeout(tib)        
        #print(f'Finished at {env.now}')
        yield pickupWindow.release(pwReq)
        
            
        

    else:
        global bailedOut
        bailedOut += 1
        #print(f'Too many in line at {len(orderWindow.queue)}, {bailedOut}')
    
    
    
# Setup and start the simulation



repetitions = range(REP)

meanArrivals = np.array([None]*STEPS)
for i in range(0, STEPS):
    meanArrivals[i] = 1+(i/10)

tempArr = np.array([None]*REP)
  
df = pd.DataFrame()
  
#print(meanArrivals)
# Start processes and run
for i in range(STEPS):
    arrivRate = meanArrivals[i]
    tempMean = 0
    for j in repetitions:
        env = simpy.Environment()
        orderLine = simpy.Resource(env, capacity=7)
        orderWindow = simpy.Resource(env, capacity=ORDER_WINDOWS)
        paymentStation = simpy.Resource(env, capacity=1)
        payLine = simpy.Resource(env, capacity=4)
        pickupLine = simpy.Resource(env, capacity=1)
        pickupWindow = simpy.Resource(env, capacity=1)
        env.process(source(env, arrivRate, orderWindow))
        random.seed(RANDOM_SEED)
        env.run(until=120)
        #print(f'ArrivalRate: {meanArrivals[i]} finished: {finishedShopping} BailedOut: {bailedOut}')
        tempArr[j] = bailedOut
        bailedOut = 0
        RANDOM_SEED += 1
        #print("Done")
    df = pd.concat([df, pd.DataFrame(tempArr, columns = [meanArrivals[i]])], axis = 1)


calcMeans = np.array([None]*STEPS)
for i in range(STEPS):
    calcMeans[i] = df[meanArrivals[i]].sum()/df.shape[0]
print(calcMeans)


plt.figure(1)
p = plt.hist(df[meanArrivals[12]], bins = 10, density = True)

bCtr = []
for i in range(len(p[1])-1):
    bCtr.append((p[1][i] + p[1][i+1]) / 2) 

bPr = p[0] / np.sum(p[0])
print(bPr)
print(np.std(df[meanArrivals[12]].values))
sampleMean = np.sum(bCtr * bPr) 
mean = 1/sampleMean
xv = np.arange(p[1][0], p[1][-1], 0.01)
yv = []
for x in xv:
    yv.append(mean * np.exp(-(mean * x)))

plt.plot(xv, yv)

plt.figure(2)

if(ORDER_WINDOWS == 1):
    df.to_excel("data1.xlsx")
else:
    df.to_excel("data2.xlsx")
plt.plot(calcMeans, meanArrivals)
plt.xlabel('Customers who left')
plt.ylabel('Minutes per customer arrival')


#print(p[1])