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

alpha=(decayAlpha(0.65,0.01,1000,"Exp"))/1000
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

epsilon=(decayEps(1,0.01,1000,"Exp"))
# print(epsilon)


def actionSelect(Q,s,epsilon):
    if random.random()>epsilon:
        a = np.argmax(Q[s])
        pos = []
        for i in range(len(Q[s])):
            if(Q[s][i]==Q[s][a]):
                pos.append(i)
        a = random.choice(pos)
    else:
        a = random.randint(0,1)
    return a
    
# Q-Learning
def q_learning(env, gamma, alpha, epsilon, noEpisodes):
    # your code goes here
    Q = np.zeros((500,2))
    Qs = np.zeros((noEpisodes,500,2))
    R = np.zeros(noEpisodes)
    actions = []
    actions.append(0)
    for e in range(noEpisodes):
        alpha0 = alpha[e]
        epsilon0 = epsilon[e]
        s,done = env.reset()
        Re = 0
        cnt = 0
        while not done:
            a = actionSelect(Q,s,epsilon0)
            if e == noEpisodes-1:
                if a==0:
                    actions.append(0)
                else:
                    actions[-1]+=1
            # print(a)
            sprime,r,done = env.step(a)
            R[e] += r
            Re += r 
            cnt += 1
            # print(Re/cnt)
            td_target = Re/(cnt)
            if not done:
                td_target = td_target + gamma*np.amax(Q[sprime])
            td_error = td_target - Q[s][a]
            Q[s][a] = Q[s][a] + alpha0*td_error
            s = sprime
        Qs[e] = Q
    V = np.amax(Q,axis = 1)
    pi = np.argmax(Q,axis = 1)
    state_value = V
    q_value = Qs
    optimal_policy = pi
    # return state-value,q-value and optimal-policy
    # state_value is a numpy array of shape [noEpisodes, states]
    return R,actions

# env = Foraging() 
# env.seed(45)   
# R = q_learning(env,1,alpha,epsilon,1000)
# # # print(q)
# print(R)
R1=[]
R2=[]
R3=[]
R4=[]

for i in range(1):
    random.seed(i+50)

    env1=Foraging(interval_time = 10,decision_time=1)
    env2=Foraging(interval_time = 10,decision_time=1.4)
    env3=Foraging(interval_time = 10,decision_time=0.6)
    env4=Foraging(interval_time = 10,decision_time=random.uniform(0.6,1.4))
    
    env1.seed(i+50)
    env2.seed(i+50)
    env3.seed(i+50)
    env4.seed(i+50)

    r1,a1=q_learning(env1,1,alpha,epsilon,1000)
    r2,a2=q_learning(env2,1,alpha,epsilon,1000)
    r3,a3=q_learning(env3,1,alpha,epsilon,1000)
    r4,a4=q_learning(env4,1,alpha,epsilon,1000)

    R1.append(r1)
    R2.append(r2)
    R3.append(r3)
    R4.append(r4)

R1=np.array(R1)
R2=np.array(R2)
R3=np.array(R3)
R4=np.array(R4)

a1=np.array(a1)
a2=np.array(a2)
a3=np.array(a3)
a4=np.array(a4)

A1=np.average(R1,axis=0)
A2=np.average(R2,axis=0)
A3=np.average(R3,axis=0)
A4=np.average(R4,axis=0)

x = np.arange(1000)
plt.plot(x,A1, color = 'blue')
plt.plot(x,A2, color = 'red')
plt.plot(x,A3, color = 'yellow')
plt.plot(x,A4, color = 'green')
plt.legend(["Decision Time = 1","Decision Time = 1.4", "Decision Time = 0.6", "Decision Time = Uniform(0.6,1.4)"])
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Q learning Rewards vs Episodes")
plt.savefig('Qlearning reward31.pdf')
plt.close()

x = np.arange(1000)
plt.plot(a1, color = 'blue')
plt.plot(a2, color = 'red')
plt.plot(a3, color = 'yellow')
plt.plot(a4, color = 'green')
plt.legend(["Decision Time = 1","Decision Time = 1.4", "Decision Time = 0.6", "Decision Time = Uniform(0.6,1.4)"])
plt.xlabel("No of Patches")
plt.ylabel("Number of harvests")
plt.title("No of harvests vs Patches")
plt.savefig('Harvest Rewards31.pdf')
plt.close()


R1=[]
R2=[]
R3=[]
R4=[]

for i in range(1):
    random.seed(i+50)

    env1=Foraging(interval_time = 3,decision_time=1)
    env2=Foraging(interval_time = 3,decision_time=1.4)
    env3=Foraging(interval_time = 3,decision_time=0.6)
    env4=Foraging(interval_time = 3,decision_time=random.uniform(0.6,1.4))
    
    env1.seed(i+50)
    env2.seed(i+50)
    env3.seed(i+50)
    env4.seed(i+50)

    r1,a1=q_learning(env1,1,alpha,epsilon,1000)
    r2,a2=q_learning(env2,1,alpha,epsilon,1000)
    r3,a3=q_learning(env3,1,alpha,epsilon,1000)
    r4,a4=q_learning(env4,1,alpha,epsilon,1000)

    R1.append(r1)
    R2.append(r2)
    R3.append(r3)
    R4.append(r4)

R1=np.array(R1)
R2=np.array(R2)
R3=np.array(R3)
R4=np.array(R4)

a1=np.array(a1)
a2=np.array(a2)
a3=np.array(a3)
a4=np.array(a4)

A1=np.average(R1,axis=0)
A2=np.average(R2,axis=0)
A3=np.average(R3,axis=0)
A4=np.average(R4,axis=0)

x = np.arange(1000)
plt.plot(x,A1, color = 'blue')
plt.plot(x,A2, color = 'red')
plt.plot(x,A3, color = 'yellow')
plt.plot(x,A4, color = 'green')
plt.legend(["Decision Time = 1","Decision Time = 1.4", "Decision Time = 0.6", "Decision Time = Uniform(0.6,1.4)"])
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Q learning Rewards vs Episodes")
plt.savefig('Qlearning reward32.pdf')
plt.close()

x = np.arange(1000)
plt.plot(a1, color = 'blue')
plt.plot(a2, color = 'red')
plt.plot(a3, color = 'yellow')
plt.plot(a4, color = 'green')
plt.legend(["Decision Time = 1","Decision Time = 1.4", "Decision Time = 0.6", "Decision Time = Uniform(0.6,1.4)"])
plt.xlabel("No of Patches")
plt.ylabel("Number of harvests")
plt.title("No of harvests vs Patches")
plt.savefig('Harvest Rewards32.pdf')
plt.close()