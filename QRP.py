import gym
from gym import spaces
import numpy as np
from numpy import random as rand
import random
import math
import matplotlib.pyplot as plt
import sys
sys.path.append('../')

from richPatch_poorPatch import ForagingRichPoorPatch 


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

alpha=(decayAlpha(0.65,0.01,1000,"Exp"))/100000
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


def actionSelect(Q,p,s,epsilon):
    if random.random()>epsilon:
        a = np.argmax(Q[p][s])
        pos = []
        for i in range(len(Q[p][s])):
            if(Q[p][s][i]==Q[p][s][a]):
                pos.append(i)
        a = random.choice(pos)
    else:
        a = random.randint(0,1)
    return a
    
# Q-Learning
def q_learning(env, gamma, alpha, epsilon, noEpisodes,P):
    # your code goes here
    Q = np.zeros((100,50,2))
    Qs = np.zeros((noEpisodes,100,50,2))
    R = np.zeros(noEpisodes)
    actions = []
    actions.append(0)
    for e in range(noEpisodes):
        alpha0 = alpha[e//5]
        epsilon0 = epsilon[e//5]
        s,done = env.reset()
        p=0
        p_prime = 0
        env.set_init_reward(P[p_prime])
        while not done:
            a = actionSelect(Q,p,s,epsilon0)
            if e == noEpisodes-1:
                if a==0:
                    actions.append(0)
                else:
                    actions[-1]+=1
            # print(a)
            sprime,r,done = env.step(a)
            if sprime==0:
                p_prime+=1
                env.set_init_reward(P[p_prime])
            R[e] += r
            td_target = r
            if not done:
                td_target = td_target + gamma*np.amax(Q[p_prime][sprime])
            td_error = td_target - Q[p][s][a]
            Q[p][s][a] = Q[p][s][a] + alpha0*td_error
            s = sprime
        Qs[e] = Q
    V = np.amax(Q,axis = 1)
    pi = np.argmax(Q,axis = 1)
    state_value = V
    q_value = Qs
    optimal_policy = pi
    # return state-value,q-value and optimal-policy
    # state_value is a numpy array of shape [noEpisodes, states]
    return R,actions,Qs

# env = Foraging() 
# env.seed(45)   
# R = q_learning(env,1,alpha,epsilon,1000)
# # # print(q)
# print(R)
# R1=[]
# R2=[]
# R3=[]
# R4=[]

# for i in range(20):
#     random.seed(i+50)
#     P = rand.choice([random.randint(2,7), random.randint(10,15)], size = 100)
#     print(P)
#     env1=ForagingRichPoorPatch(interval_time = 10,decision_time=1)
#     # env2=ForagingRichPoorPatch(interval_time = 10,decision_time=1.4)
#     # env3=ForagingRichPoorPatch(interval_time = 10,decision_time=0.6)
#     env4=ForagingRichPoorPatch(interval_time = 10,decision_time=random.uniform(0.6,1.4))
    
#     env1.seed(i+50)
#     # env2.seed(i+50)
#     # env3.seed(i+50)
#     env4.seed(i+50)

#     r1,a1,q1=q_learning(env1,1,alpha,epsilon,5000,P)
#     # r2,a2,q2=q_learning(env2,1,alpha,epsilon,5000)
#     # r3,a3,q3=q_learning(env3,1,alpha,epsilon,5000)
#     r4,a4,q4=q_learning(env4,1,alpha,epsilon,5000,P)

#     R1.append(r1)
#     # R2.append(r2)
#     # R3.append(r3)
#     R4.append(r4)

# # print(q1.shape)
# # print(q1.shape)
# # d1 = np.zeros((1000,10))
# # for i in range(1000):
# #     d1[i][0] += q1[i][0][1]
# #     d1[i][1] += q1[i][1][1] 
# #     d1[i][2] += q1[i][2][1] 
# #     d1[i][3] += q1[i][3][1] 
# #     d1[i][4] += q1[i][4][1] 
# #     d1[i][5] += q1[i][5][1] 
# #     d1[i][6] += q1[i][6][1] 
# #     d1[i][7] += q1[i][7][1] 
# #     d1[i][8] += q1[i][8][1] 
# #     d1[i][9] += q1[i][9][1] 

# # print(d1.shape)
# # d1 = abs(47.5-np.sum(d1,axis=1))
# # print(d1.shape)
# # plt.plot(d1[:])
# # plt.legend()
# # plt.show()

# R1=np.array(R1)
# # R2=np.array(R2)
# # R3=np.array(R3)
# R4=np.array(R4)

# a1=np.array(a1)
# # a2=np.array(a2)
# # a3=np.array(a3)
# a4=np.array(a4)

# A1=np.average(R1,axis=0)
# # A2=np.average(R2,axis=0)
# # A3=np.average(R3,axis=0)
# A4=np.average(R4,axis=0)

# x = np.arange(5000)
# plt.plot(x,A1, color = 'blue')
# # plt.plot(x,A2, color = 'red')
# # plt.plot(x,A3, color = 'yellow')
# plt.plot(x,A4, color = 'green')
# plt.legend(["Decision Time = 1", "Decision Time = Uniform(0.6,1.4)"])
# plt.xlabel("Episodes")
# plt.ylabel("Rewards")
# plt.title("Q learning Rewards vs Episodes")
# plt.savefig('Qlearning reward101.pdf')
# plt.close()

# x = np.arange(5000)
# plt.plot(a1, color = 'blue')
# # plt.plot(a2, color = 'red')
# # plt.plot(a3, color = 'yellow')
# plt.plot(a4, color = 'green')
# plt.legend(["Decision Time = 1", "Decision Time = Uniform(0.6,1.4)"])
# plt.xlabel("No of Patches")
# plt.ylabel("Number of harvests")
# plt.title("No of harvests vs Patches")
# plt.savefig('Harvest Rewards1011.pdf')
# plt.close()


R1=[]
R2=[]
R3=[]
R4=[]

for i in range(20):
    random.seed(i+50)
    rand.seed(i+50)
    P = rand.choice([random.randint(2,7), random.randint(10,15)], size = 100)
    print(P)
    env1=ForagingRichPoorPatch(interval_time = 3,decision_time=1)
    # env2=Foraging(interval_time = 3,decision_time=1.4)
    # env3=Foraging(interval_time = 3,decision_time=0.6)
    env4=ForagingRichPoorPatch(interval_time = 3,decision_time=random.uniform(0.6,1.4))
    
    env1.seed(i+50)
    # env2.seed(i+50)
    # env3.seed(i+50)
    env4.seed(i+50)

    r1,a1,q1=q_learning(env1,1,alpha,epsilon,5000,P)
    # r2,a2=q_learning(env2,1,alpha,epsilon,1000)
    # r3,a3=q_learning(env3,1,alpha,epsilon,1000)
    r4,a4,q1=q_learning(env4,1,alpha,epsilon,5000,P)

    R1.append(r1)
    # R2.append(r2)
    # R3.append(r3)
    R4.append(r4)

R1=np.array(R1)
# R2=np.array(R2)
# R3=np.array(R3)
R4=np.array(R4)

a1=np.array(a1)
# a2=np.array(a2)
# a3=np.array(a3)
a4=np.array(a4)

A1=np.average(R1,axis=0)
# A2=np.average(R2,axis=0)
# A3=np.average(R3,axis=0)
A4=np.average(R4,axis=0)

# x = np.arange(1000)
plt.plot(A1[0:3000], color = 'blue')
# plt.plot(x,A2, color = 'red')
# plt.plot(x,A3, color = 'yellow')
plt.plot(A4[0:3000], color = 'green')
plt.legend(["Decision Time = 1", "Decision Time = Uniform(0.6,1.4)"])
plt.xlabel("Episodes")
plt.ylabel("Rewards")
plt.title("Q learning Rewards vs Episodes")
plt.savefig('Qlearning reward102.pdf')
plt.close()

x = np.arange(1000)
plt.plot(a1, color = 'blue')
# plt.plot(a2, color = 'red')
# plt.plot(a3, color = 'yellow')
plt.plot(a4, color = 'green')
plt.legend(["Decision Time = 1", "Decision Time = Uniform(0.6,1.4)"])
plt.xlabel("No of Patches")
plt.ylabel("Number of harvests")
plt.title("No of harvests vs Patches")
plt.savefig('Harvest Rewards102.pdf')
plt.close()