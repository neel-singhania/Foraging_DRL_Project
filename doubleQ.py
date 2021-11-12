import gym
from gym import spaces
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import sys 
sys.path.append('../')

from foraging import Foraging

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

alpha=(decayAlpha(0.5,0.1,1000,"Exp"))
# print(alpha)

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

epsilon=(decayEps(0.5,0.0001,1000,"Exp"))
# print(epsilon)
def actionSelect(s,Q,E,h):
    if random.random()>E:
        a = np.argmax(Q[s][h])
    else:
        a = 1-np.argmax(Q[s][h])
    # print(a)
    return a
    
def DoubleQLearning(env, gamma, alpha, epsilon, noEepisodes,states,harvests):
    Q1 = np.zeros((states,harvests,2))
    Qs1 = np.zeros((noEepisodes,states,harvests,2))
    Q = np.zeros((states,harvests,2))
    Qs = np.zeros((noEepisodes,states,harvests,2))
    Q2 = np.zeros((states,harvests,2))
    Qs2 = np.zeros((noEepisodes,states,harvests,2))
    R = np.zeros(noEepisodes)
    for e in range(noEepisodes):
        alp = alpha[e]
        E = epsilon[e]
        # print(E)
        s,done = env.reset()
        s = 0
        h = 0
        while not done:
            a = actionSelect(s,Q,E,h)
            # print(a)
            hnext,r,done,info = env.step(a)
            # print(a,h,r,done,s,alp,E)
            if hnext==0:
                snext = s+1
            else:
                snext = s
            R[e] += r
            td_target = r
            if random.randint(0,1):
                aq1 = np.argmax(Q1[snext][hnext])
                if not done:
                    td_target =td_target+gamma*Q2[snext][hnext][aq1]
                td_error = td_target - Q1[s][h][a]
                Q1[s][h][a] = Q1[s][h][a]+alp*td_error
            else:
                aq2 = np.argmax(Q2[snext][hnext])
                if not done:
                    td_target =td_target+gamma*Q1[snext][hnext][aq2]
                td_error = td_target - Q2[s][h][a]
                Q2[s][h][a] = Q1[s][h][a]+alp*td_error
            
            # s = sp
            s = snext
            h = hnext
        
        Qs1[e] = Q1
        Qs2[e] = Q2
        Q = (Q1+Q2)/2
        Qs[e] = Q
        
        
    # V = max(Q,axis = 1)
    return R

env = Foraging() 
env.seed(45)   
r = DoubleQLearning(env,1,alpha,epsilon,1000,35,20)
# # print(q)
print(r)
# R=[]
# for i in range(5):
#     env=Foraging()
#     env.seed(i+50)
#     random.seed(i+50)
#     r=QLearning(env,1,alpha,epsilon,10,30,20)
#     R.append(r)
# R=np.array(R)
# A=np.average(R,axis=0)
# x = np.arange(1000)
# plt.plot(x,A)
# plt.xlabel("Episodes")
# plt.ylabel("Rewards")
# plt.title("Q learning Rewards vs Episodes")
# plt.savefig('Qlearning reward.pdf')
# plt.close()