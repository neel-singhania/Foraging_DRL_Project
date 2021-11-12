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

alpha=(decayAlpha(0.75,0.01,1000,"Exp"))/500
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
def double_q_learning(env, gamma, alpha, epsilon, noEpisodes):
    # your code goes here
    Q1 = np.zeros((500,2))
    Qs1 = np.zeros((noEpisodes,500,2))
    Q2 = np.zeros((500,2))
    Qs2 = np.zeros((noEpisodes,500,2))
    Q = np.zeros((500,2))
    Qs = np.zeros((noEpisodes,500,2))
    Vs = np.zeros((noEpisodes,500))
    Ps = np.zeros((noEpisodes,500))
    R = np.zeros(noEpisodes)
    actions = []
    actions.append(0)
    for e in range(noEpisodes):
        alpha0 = alpha[e]
        epsilon0 = epsilon[e]
        s,done = env.reset()
        while not done:
            a = actionSelect(Q,s,epsilon0)
            if e == noEpisodes-1:
                if a==0:
                    actions.append(0)
                else:
                    actions[-1]+=1
            sprime,r,done = env.step(a)
            R[e] += r
            if np.random.randint(0,2):
                aq1 = np.argmax(Q1[sprime])
                td_target = r
                if not done:
                    td_target = td_target + gamma*(Q2[sprime][aq1])
                td_error = td_target - Q1[s][a]
                Q1[s][a] = Q1[s][a] + alpha0*td_error
            else:
                aq2 = np.argmax(Q2[sprime])
                td_target = r
                if not done:
                    td_target = td_target + gamma*(Q1[sprime][aq2])
                td_error = td_target - Q2[s][a]
                Q2[s][a] = Q1[s][a] + alpha0*td_error
            s = sprime
        #Book-Keeping
        Qs1[e] = Q1
        Qs2[e] = Q2
        Q = (Q1+Q2)/2
        Qs[e] = Q
        # Vs[e] = np.amax(Q,axis = 1)
        # Ps[e] = np.argmax(Q,axis = 1)
        
    # V = np.amax(Q,axis = 1)
    # pi = np.argmax(Q,axis = 1)
    # state_value = Vs
    # q_value = Qs
    # optimal_policy = Ps
    # return state-value,q-value and optimal-policy
    # state_value is a numpy array of shape [noEpisodes, states]
    return R,actions,Qs
# env = Foraging() 
# env.seed(45)   
# R = q_learning(env,1,alpha,epsilon,1000)
# # # print(q)
# print(R)
R1=[]
R2=[]
R3=[]
R4=[]

Ac1 = []
Ac2 = []
Ac3 = []
Ac4 = []


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

    r1,a1,q1=double_q_learning(env1,1,alpha,epsilon,1000)
    r2,a2,q1=double_q_learning(env2,1,alpha,epsilon,1000)
    r3,a3,q1=double_q_learning(env3,1,alpha,epsilon,1000)
    r4,a4,q1=double_q_learning(env4,1,alpha,epsilon,1000)

    R1.append(r1)
    R2.append(r2)
    R3.append(r3)
    R4.append(r4)

    # Ac1.append(a1)
    # Ac2.append(a2)
    # Ac3.append(a3)
    # Ac4.append(a4)

print(q1.shape)
d1 = np.zeros((1000,10))
for i in range(1000):
    d1[i][0] += q1[i][0][1]
    d1[i][1] += q1[i][1][1] 
    d1[i][2] += q1[i][2][1] 
    d1[i][3] += q1[i][3][1] 
    d1[i][4] += q1[i][4][1] 
    d1[i][5] += q1[i][5][1] 
    d1[i][6] += q1[i][6][1] 
    d1[i][7] += q1[i][7][1] 
    d1[i][8] += q1[i][8][1] 
    d1[i][9] += q1[i][9][1] 

print(d1.shape)
d1 = abs(47.5-np.sum(d1,axis=1))
print(d1.shape)
plt.plot(d1[:])
plt.legend()
plt.show()
R1=np.array(R1)
R2=np.array(R2)
R3=np.array(R3)
R4=np.array(R4)

# Ac1=np.array(Ac1)
# Ac2=np.array(Ac2)
# Ac3=np.array(Ac3)
# Ac4=np.array(Ac4)

A1=np.average(R1,axis=0)
A2=np.average(R2,axis=0)
A3=np.average(R3,axis=0)
A4=np.average(R4,axis=0)

# Ac1=np.around(np.average(Ac1,axis = 0))
# Ac2=np.around(np.average(Ac2,axis = 0))
# Ac3=np.around(np.average(Ac3,axis = 0))
# Ac4=np.around(np.average(Ac4,axis = 0))

# x = np.arange(1000)
# plt.plot(x,A1, color = 'blue')
# plt.plot(x,A2, color = 'red')
# plt.plot(x,A3, color = 'yellow')
# plt.plot(x,A4, color = 'green')
# plt.legend(["Decision Time = 1","Decision Time = 1.4", "Decision Time = 0.6", "Decision Time = Uniform(0.6,1.4)"])
# plt.xlabel("Episodes")
# plt.ylabel("Rewards")
# plt.title("Q learning Rewards vs Episodes")
# plt.savefig('Qlearning reward21.pdf')
# plt.close()

# x = np.arange(1000)
# plt.plot(a1, color = 'blue')
# plt.plot(a2, color = 'red')
# plt.plot(a3, color = 'yellow')
# plt.plot(a4, color = 'green')
# plt.legend(["Decision Time = 1","Decision Time = 1.4", "Decision Time = 0.6", "Decision Time = Uniform(0.6,1.4)"])
# plt.xlabel("No of Patches")
# plt.ylabel("Number of harvests")
# plt.title("No of harvests vs Patches")
# plt.savefig('Harvest Rewards21.pdf')
# plt.close()

# R1=[]
# R2=[]
# R3=[]
# R4=[]

# Ac1 = []
# Ac2 = []
# Ac3 = []
# Ac4 = []


# for i in range(50):
#     random.seed(i+50)

#     env1=Foraging(interval_time = 3,decision_time=1)
#     env2=Foraging(interval_time = 3,decision_time=1.4)
#     env3=Foraging(interval_time = 3,decision_time=0.6)
#     env4=Foraging(interval_time = 3,decision_time=random.uniform(0.6,1.4))
    
#     env1.seed(i+50)
#     env2.seed(i+50)
#     env3.seed(i+50)
#     env4.seed(i+50)

#     r1,a1=double_q_learning(env1,1,alpha,epsilon,1000)
#     r2,a2=double_q_learning(env2,1,alpha,epsilon,1000)
#     r3,a3=double_q_learning(env3,1,alpha,epsilon,1000)
#     r4,a4=double_q_learning(env4,1,alpha,epsilon,1000)

#     R1.append(r1)
#     R2.append(r2)
#     R3.append(r3)
#     R4.append(r4)

#     # Ac1.append(a1)
#     # Ac2.append(a2)
#     # Ac3.append(a3)
#     # Ac4.append(a4)

# R1=np.array(R1)
# R2=np.array(R2)
# R3=np.array(R3)
# R4=np.array(R4)

# # Ac1=np.array(Ac1)
# # Ac2=np.array(Ac2)
# # Ac3=np.array(Ac3)
# # Ac4=np.array(Ac4)

# A1=np.average(R1,axis=0)
# A2=np.average(R2,axis=0)
# A3=np.average(R3,axis=0)
# A4=np.average(R4,axis=0)

# # Ac1=np.around(np.average(Ac1,axis = 0))
# # Ac2=np.around(np.average(Ac2,axis = 0))
# # Ac3=np.around(np.average(Ac3,axis = 0))
# # Ac4=np.around(np.average(Ac4,axis = 0))

# x = np.arange(1000)
# plt.plot(x,A1, color = 'blue')
# plt.plot(x,A2, color = 'red')
# plt.plot(x,A3, color = 'yellow')
# plt.plot(x,A4, color = 'green')
# plt.legend(["Decision Time = 1","Decision Time = 1.4", "Decision Time = 0.6", "Decision Time = Uniform(0.6,1.4)"])
# plt.xlabel("Episodes")
# plt.ylabel("Rewards")
# plt.title("Q learning Rewards vs Episodes")
# plt.savefig('Qlearning reward22.pdf')
# plt.close()

# x = np.arange(1000)
# plt.plot(a1, color = 'blue')
# plt.plot(a2, color = 'red')
# plt.plot(a3, color = 'yellow')
# plt.plot(a4, color = 'green')
# plt.legend(["Decision Time = 1","Decision Time = 1.4", "Decision Time = 0.6", "Decision Time = Uniform(0.6,1.4)"])
# plt.xlabel("No of Patches")
# plt.ylabel("Number of harvests")
# plt.title("No of harvests vs Patches")
# plt.savefig('Harvest Rewards22.pdf')
# plt.close()

