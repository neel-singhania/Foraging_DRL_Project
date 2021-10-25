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
params["noEpisodes"]=1000
params["maxHarvest"]=30
params["maxPatches"]=25

def decayAlpha(initialValue, finalValue, maxSteps, decayType):
    step_size=np.zeros(maxSteps)
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

alpha0=(decayAlpha(0.5,0.0001,params["noEpisodes"],"Exp"))

def decayEps(initialValue, finalValue, maxSteps, decayType):
    step_size=np.zeros(maxSteps)
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

epsilon0=(decayEps(0.7,0.01,params["noEpisodes"],"Exp"))

random.seed(40)
env=Foraging()
env.seed(40)

def actionSelect(p,h,Q,epsilon):
    if random.random()>epsilon:
        a=np.argmax(Q[p][h])
    else:
        a=1-np.argmax(Q[p][h])
    return a

# print(actionSelect(0,0,Q,0.7))

def Sarsa(env,gamma,alpha0,epsilon0,noEpisodes):
    #initialisation
    Q=np.zeros((params["maxPatches"],params["maxHarvest"],env.action_space.n))
    Qs=np.zeros((noEpisodes,params["maxPatches"],params["maxHarvest"],env.action_space.n))
    #Algorithm
    R=np.zeros((noEpisodes))
    for e in range (noEpisodes):
        env.reset()
        alpha=alpha0[e]
        epsilon=epsilon0[e]
        p,h,done=0,0,False
        a=actionSelect(p,h,Q,epsilon)
        # print(t)
        reward=0
        while not done:
            h_next,r,done,info=env.step(a)
            reward+=r
            p_next=p
            if h_next==0:
                p_next=p+1
            a_next=actionSelect(p_next,h_next,Q,epsilon)
            td_target=r
            if not done:
                td_target=td_target+gamma*Q[p_next][h_next][a_next]
            td_error=td_target-Q[p][h][a]
            Q[p][h][a]=Q[p][h][a]+alpha*td_error
            p,h,a=p_next,h_next,a_next
        R[e]=reward #bookkeeping for rewards
        Qs[e]=Q
    V,pi=np.amax(Q,axis=2),np.argmax(Q,axis=2)
    return R

R=[]
for i in range(50):
    env=Foraging()
    env.seed(i+50)
    random.seed(i+50)
    r=Sarsa(env,1,alpha0,epsilon0,params["noEpisodes"],)
    R.append(r)
R=np.array(R)
A=np.average(R,axis=0)
x = np.arange(params["noEpisodes"])
plt.plot(x,A)
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Sarsa Rewards vs Episodes")
plt.savefig('Sarsa reward.pdf')
plt.close()