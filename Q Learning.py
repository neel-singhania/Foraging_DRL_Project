import gym
from gym import spaces
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import sys 
sys.path.append('../')

from foraging import Foraging


params={}
params["noEpisodes"]=2000
params["nodecayEpisodes"]=1000 
params["maxHarvest"]=30
params["maxPatches"]=80

def decayAlpha(initialValue, finalValue, maxSteps, decayType):
    step_size=np.zeros(2000)
    if decayType=="Linear":
        decayRate=(initialValue-finalValue)/maxSteps
        for i in range(maxSteps):
            step_size[i]=initialValue-i*decayRate
    if decayType=="Exp":
        decayRate=(np.log(initialValue/finalValue))/maxSteps
        step_size[0]=initialValue
        for i in range(maxSteps-1):
            step_size[i+1]=step_size[i]*np.exp(-decayRate)
    return step_size

alpha0=(decayAlpha(0.5,0.1,params["nodecayEpisodes"],"Exp"))



def decayEps(initialValue, finalValue, maxSteps, decayType):
    step_size=np.zeros(2000)
    if decayType=="Linear":
        decayRate=(initialValue-finalValue)/maxSteps
        for i in range(maxSteps):
            step_size[i]=initialValue-i*decayRate
    if decayType=="Exp":
        decayRate=(np.log(initialValue/finalValue))/maxSteps
        step_size[0]=initialValue
        for i in range(maxSteps-1):
            step_size[i+1]=step_size[i]*np.exp(-decayRate)
    return step_size

epsilon0=(decayEps(0.5,0.001,params["nodecayEpisodes"],"Exp"))

for i in range(1000):
    alpha0[i+1000]=0.1
    epsilon0[i+1000]=0.001

# print(alpha0)

def actionSelect(p,h,Q,epsilon):
    if random.random()>epsilon:
        a=np.argmax(Q[p][h])
    else:
        a=1-np.argmax(Q[p][h])
    return a

def Qlearning(env,gamma,alpha0,epsilon0,noEpisodes):
    Q=np.zeros((params["maxPatches"],params["maxHarvest"],env.action_space.n))
    Qs=np.zeros((params["noEpisodes"],params["maxPatches"],params["maxHarvest"],env.action_space.n))
    R=[]
    for e in range(noEpisodes):
        alpha=alpha0[e]
        epsilon=epsilon0[e]
        env.reset()
        reward=0
        p,h,done=0,0,False
        while not done:
            a=actionSelect(p,h,Q,epsilon)
            h_next,r,done,info=env.step(a)
            p_next=p
            if h_next==0:
                p_next=p+1
            reward+=r
            td_target=r
            if not done:
                td_target=td_target+gamma*np.amax(Q[p][h])
            td_error=td_target-Q[p][h][a]
            Q[p][h][a]=Q[p][h][a]+alpha*td_error
            p,h=p_next,h_next
        R.append(reward) 
        Qs[e]=Q
    
    V,pi=np.amax(Q,axis=2),np.argmax(Q,axis=2)
    return R

# env=Foraging()
# env.seed(50)
# random.seed(50)
# print(Qlearning(env,1,alpha0,epsilon0,params["noEpisodes"]))

R=[]
for i in range(50):
    env=Foraging(interval_time = 3)
    env.seed(i+50)
    random.seed(i+50)
    r=Qlearning(env,1,alpha0,epsilon0,params["noEpisodes"])
    R.append(r)
R=np.array(R)
A=np.average(R,axis=0)
x = np.arange(params["noEpisodes"])
plt.plot(A)
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Qlearning Rewards vs Episodes")
plt.savefig('Qlearning reward2.pdf')
plt.close()

