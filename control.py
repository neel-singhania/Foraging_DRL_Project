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
params["epsilon"]=0.3
params["initial_epsilon"]=1.0
params["final_epsilon"]={}
params["decay_rate"]={}
params["final_epsilon"]['lin']=0.0
params["final_epsilon"]['exp']=0.01
params["maxHarvest"]=15
params["maxPatches"]=24

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

epsilon0=(decayEps(0.5,0.0001,params["noEpisodes"],"Exp"))

random.seed(40)
env=Foraging()
env.seed(40)

def generateTrajectory(env,Q,epsilon):
    done=False
    p,h=0,0
    t=[]
    while not done and p<24:
        if random.random()>epsilon:
            a=np.argmax(Q[p][h])
            p_next=p
            h_next,r,done,info=env.step(a)

        else:
            a=1-np.argmax(Q[p][h])
            p_next=p
            h_next,r,done,info=env.step(a)
        if h_next==0:
            p_next=p+1
        t.append((p,h,a,r,p_next,h_next))
        p=p_next
        h=h_next
    return t

# print(generateTrajectory(env,np.zeros((params["maxPatches"],params["maxHarvest"],env.action_space.n)),0.7))

def MCcontrol(env,gamma,alpha0,epsilon0,noEpisodes,firstVisit):
    #initialisation
    Q=np.zeros((params["maxPatches"],params["maxHarvest"],env.action_space.n))
    Qs=np.zeros((noEpisodes,params["maxPatches"],params["maxHarvest"],env.action_space.n))
    #Algorithm
    R=np.zeros((noEpisodes))
    for e in range (noEpisodes):
        env.reset()
        alpha=alpha0[e]
        epsilon=epsilon0[e]
        
        t=generateTrajectory(env,Q,epsilon)
        # print(t)
        r=0
        visited=np.zeros((params["maxPatches"],params["maxHarvest"],env.action_space.n),dtype=bool)
        for i in range(len(t)):
            p=t[i][0]
            h=t[i][1]
            # print(h)
            a=t[i][2]
            r+=t[i][3]
            if visited[p][h][a] and firstVisit:
                continue
            visited[p][h][a] =True
            G=0
            for j in range(i,len(t)):
                # print(np.shape(t))
                # print(j)
                G+=(pow(gamma,j-i)*t[j][3])
            Q[p][h][a]=Q[p][h][a]+alpha*(G-Q[p][h][a])
        R[e]=r #bookkeeping for rewards
        Qs[e]=Q
    V,pi=np.amax(Q,axis=2),np.argmax(Q,axis=2)
    return R

R=[]
for i in range(5):
    env=Foraging()
    env.seed(i+50)
    random.seed(i+50)
    r=MCcontrol(env,1,alpha0,epsilon0,params["noEpisodes"],1)
    R.append(r)
R=np.array(R)
A=np.average(R,axis=0)
x = np.arange(params["noEpisodes"])
plt.plot(x,A)
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("MC Control Rewards vs Episodes")
plt.savefig('MCcontrol reward.pdf')
plt.close()