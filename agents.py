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
params["maxEpisodes"]=10
params["epsilon"]=0.3
params["initial_epsilon"]=1.0
params["final_epsilon"]={}
params["decay_rate"]={}
params["final_epsilon"]['lin']=0.0
params["final_epsilon"]['exp']=0.01
params["decay_rate"]['lin']=(params["initial_epsilon"]-params["final_epsilon"]['lin'])/params["maxEpisodes"]
params["decay_rate"]['exp']=np.log(params["initial_epsilon"]/params["final_epsilon"]['exp'])/params["maxEpisodes"]
params["c_UCB"]=0.1
params["init_temp"]=10000000
params["final_temp"]=0.05
params["temp_decay_rate_lin"]=(params["init_temp"]-params["final_temp"])/params["maxEpisodes"]
params["temp_decay_rate_exp"]=np.log(params["init_temp"]/params["final_temp"])/params["maxEpisodes"]
params["no_envs"]=50
params["decay_type_decaying_epsilon"]="lin"
params["decay_type_Softmax_exploration"]="exp"
params["arms"]=np.arange(1,16)

env=Foraging()
# env.reset()
# end_state, reward, done, info = env.step(1)
# while(reward>0):
#     end_state, reward, done, info = env.step(1)
# print(end_state)
# env.reset()
envs=[]
for i in range(50):
    envs.append(Foraging())
    envs[i].seed(i+50)

def PureExploitation(env,params):
    # print("PureExploitation")
    Q = np.zeros(len(params["arms"]))
    N = np.zeros(len(params["arms"]))
    e = 0
    Q_est = np.zeros((params["maxEpisodes"],len(params["arms"])))
    R=np.zeros((params["maxEpisodes"])-1)
    # actions=np.zeros((params["maxEpisodes"]))
    h = []
    env.reset()
    while e < params["maxEpisodes"]-1 :
        max_indices=np.where(Q==np.amax(Q))
        harvest = random.choice(max_indices[0])
        # harvest=9
        done=False
        r=0
        # print(harvest)
        while not done:
            for i in range (harvest):
                if not done:
                    end_state, reward, done, info = env.step(1)
                    # print(reward) 
                    r+=reward
            if not done:
                end_state, reward, done, info = env.step(0)
              
        # print(r)
        N[harvest-1] = N[harvest-1] + 1
        Q[harvest-1] = Q[harvest-1] + (r-Q[harvest-1])/N[harvest-1]
        
        
        R[e]=r
        e = e+1
        Q_est[e] = Q
        env.reset()
    return R

# print(PureExploitation(env,params))
def PureExploration(env,params):
    # print("PureExploration")
    Q = np.zeros(len(params["arms"]))
    N = np.zeros(len(params["arms"]))
    e = 0
    Q_est = np.zeros((params["maxEpisodes"],len(params["arms"])))
    R=np.zeros((params["maxEpisodes"])-1)
    # actions=np.zeros((params["maxEpisodes"]))
    env.reset()
    while e < params["maxEpisodes"]-1 :
        harvest= random.choice(np.arange(1,len(Q)+1))
        # print(harvest)
        done=False
        r=0
        while not done:
            for i in range (harvest):
                if not done:
                    end_state, reward, done, info = env.step(1)
                    r+=reward
            if not done:
                end_state, reward, done, info = env.step(0)
        #     print(r,reward)   
        # print(r)
        N[harvest-1] = N[harvest-1] + 1
        Q[harvest-1] = Q[harvest-1] + (r-Q[harvest-1])/N[harvest-1]
        
        
        R[e]=r
        e = e+1
        Q_est[e] = Q
        env.reset()
    return R
# print(PureExploration(env,params))
def epsilonGreedy(env,params):
    # print("epsilonGreedy")
    Q = np.zeros(len(params["arms"]))
    N = np.zeros(len(params["arms"]))
    e = 0
    Q_est = np.zeros((params["maxEpisodes"],len(params["arms"])))
    R=np.zeros((params["maxEpisodes"])-1)
    # actions=np.zeros((params["maxEpisodes"]))
    env.reset()
    while e < params["maxEpisodes"]-1 :
        if random.random() > params["epsilon"]:
            max_indices=np.where(Q==np.amax(Q))
            harvest = random.choice(max_indices[0])
        else :
            harvest= random.choice(np.arange(1,len(Q)+1))
        # print(harvest)
        done=False
        r=0
        while not done:
            for i in range (harvest):
                if not done:
                    end_state, reward, done, info = env.step(1)
                    r+=reward
            if not done:
                end_state, reward, done, info = env.step(0)
        #     print(r,reward)   
        # print(r)
        N[harvest-1] = N[harvest-1] + 1
        Q[harvest-1] = Q[harvest-1] + (r-Q[harvest-1])/N[harvest-1]
        
        
        R[e]=r
        e = e+1
        Q_est[e] = Q
        env.reset()
    return R
# print(epsilonGreedy(env,params))
def decayingEpsilonGreedy(env,params,type):
    # print("decayingEpsilonGreedy")
    Q = np.zeros(16)
    N = np.zeros(16)
    e = 0
    Q_est = np.zeros((params["maxEpisodes"],16))
    R=np.zeros((params["maxEpisodes"])-1)
    # actions=np.zeros((params["maxEpisodes"]))
    env.reset()
    epsilon=params["initial_epsilon"]
    while e < params["maxEpisodes"]-1 :
        if random.random() > epsilon:
            max_indices=np.where(Q==np.amax(Q))
            # print(epsilon,e,max_indices[0])
            harvest = random.choice(max_indices[0])
        else :
            harvest= random.choice(np.arange(1,len(Q)))
        # print(harvest)
        done=False
        r=0
        while not done:
            for i in range (harvest):
                if not done:
                    end_state, reward, done, info = env.step(1)
                    r+=reward
            if not done:
                end_state, reward, done, info = env.step(0)
        #     print(r,reward)   
        # print(r)
        N[harvest] = N[harvest] + 1
        Q[harvest] = Q[harvest] + (r-Q[harvest])/N[harvest]
        
        if type=='lin':
            epsilon=epsilon-params["decay_rate"]['lin']
        else :
            epsilon = epsilon*np.exp(-params["decay_rate"]['exp'])
            
        R[e]=r
        e = e+1
        Q_est[e] = Q
        env.reset()
    return R
# print(decayingEpsilonGreedy(env,params,"exp"))
def UCBexploration(env,params):
    # print("UCBexploration")
    Q = np.zeros(17)
    N = np.ones(17)
    e = 0
    Q_est = np.zeros((params["maxEpisodes"],17))
    R=np.zeros((params["maxEpisodes"])-1)
    # actions=np.zeros((params["maxEpisodes"]))
    env.reset()
    while e < params["maxEpisodes"] - 1:
        if e< 16:
            harvest = e+1
        else:
            # if e==51:
            #     print(N)
            U= params["c_UCB"]* math.sqrt(math.log(e))/np.sqrt(N)
            UCB = np.add(Q,U)
            max_indices=np.where(UCB==np.amax(UCB))
            harvest = random.choice(max_indices[0])
            N[harvest] = N[harvest] + 1
        # print(harvest)
        done=False
        r=0
        while not done:
            for i in range (harvest):
                if not done:
                    end_state, reward, done, info = env.step(1)
                    r+=reward
            if not done:
                end_state, reward, done, info = env.step(0)
        # print(r)
        Q[harvest] = Q[harvest] + (r-Q[harvest])/N[harvest]
        R[e] =r
        e = e+1
        Q_est[e] = Q
        
        
        env.reset()
    return R
print(UCBexploration(env,params))

# exploit = []
# explore = []
# epsilon = []
# depsilon = []
# soft_max = []
# ucb = []
# for i in range(50):
#     env=Foraging(interval_time = 10)
#     env.seed(i+50)
#     random.seed(i+50)
#     r1 = PureExploitation(env,params)
#     r2 = PureExploration(env,params)
#     r3 = epsilonGreedy(env,params)
#     r4 = decayingEpsilonGreedy(env,params,"exp")
#     r6 = UCBexploration(env,params)
#     # print(np.shape(r1))
#     exploit.append(r1)
#     explore.append(r2)
#     epsilon.append(r3)
#     depsilon.append(r4)
#     ucb.append(r6)

# exploit = np.array(exploit)
# # print(np.shape(exploit))
# explore = np.array(explore)
# epsilon = np.array(epsilon)
# depsilon = np.array(depsilon)

# ucb = np.array(ucb)

# avg1 = np.average(exploit,axis=0)
# avg2 = np.average(explore,axis=0)
# avg3 = np.average(epsilon,axis=0)
# avg4 = np.average(depsilon,axis=0)
# avg6 = np.average(ucb,axis=0)
# x = np.arange(params["maxEpisodes"]-1)

# # print(np.shape(avg1))

# plt.plot(x,avg1)
# plt.xlabel("Episodes")
# plt.ylabel("Average Reward")
# plt.title("Exploitation: Average Reward vs Episodes")
# plt.savefig('exploit.png')
# plt.show()
# plt.close()

# plt.plot(x,avg2)
# plt.savefig('explore.png')
# plt.xlabel("Episodes")
# plt.ylabel("Average Reward")
# plt.title("Exploration:Average Reward vs Episodes")
# plt.show()
# plt.close()

# plt.plot(x,avg3)
# plt.savefig('epsilon.png')
# plt.xlabel("Episodes")
# plt.ylabel("Average Reward")
# plt.title("Epsilon-Greedy: Average Reward vs Episodes")
# plt.show()
# plt.close()

# plt.plot(x,avg4)
# plt.savefig('depsilon.png') 
# plt.xlabel("Episodes")
# plt.ylabel("Average Reward")
# plt.title("decay-Epsilon Greedy: Average Reward vs Episodes")
# plt.show()
# plt.close()


# plt.plot(x,avg6)
# plt.savefig('ucb.png')
# plt.xlabel("Episodes")
# plt.ylabel("Average Reward")
# plt.title("UCB: Average Reward vs Episodes")
# plt.show()
# plt.close()

# plt.plot(x,avg1,'b')
# plt.plot(x,avg2,'r')
# plt.plot(x,avg3,'y')
# plt.plot(x,avg4,'m')
# plt.plot(x,avg6,'g')
# plt.title("Average rewards received vs different agents")
# plt.xlabel("Epsiodes")
# plt.ylabel("Average rewards received")
# plt.legend(["Pure Exploitation","Pure Exploration","Epsilon Greedy","decay-Epsilon Greedy","UCB"], loc = "lower right")
# plt.savefig("3 Average rewards received vs different agents.pdf")
# plt.show()
# plt.close()
