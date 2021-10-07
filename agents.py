import gym
from gym import spaces
import numpy as np
import random
import sys 
sys.path.append('../')

from foraging import Foraging


params={}
params["maxEpisodes"]=50
params["epsilon"]=0.1
params["initial_epsilon"]=1.0
params["final_epsilon"]={}
params["decay_rate"]={}
params["final_epsilon"]['lin']=0.0
params["final_epsilon"]['exp']=0.5
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
params["arms"]=np.arange(1,50)

env=Foraging()

def PureExploitation(env,params):
    # print("PureExploitation")
    Q = np.zeros(len(params["arms"]))
    N = np.zeros(len(params["arms"]))
    e = 0
    Q_est = np.zeros((params["maxEpisodes"],len(params["arms"])))
    R=np.zeros((params["maxEpisodes"]))
    # actions=np.zeros((params["maxEpisodes"]))
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
    R=np.zeros((params["maxEpisodes"]))
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
    R=np.zeros((params["maxEpisodes"]))
    # actions=np.zeros((params["maxEpisodes"]))
    env.reset()
    while e < params["maxEpisodes"]-1 :
        if random.random() > params["epsilon"]:
            max_indices=np.where(Q==np.amax(Q))
            harvest = random.choice(max_indices[0])
        else :
            harvest= random.choice(np.arange(1,len(Q)+1))
        print(harvest)
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
    Q = np.zeros(len(params["arms"]))
    N = np.zeros(len(params["arms"]))
    e = 0
    Q_est = np.zeros((params["maxEpisodes"],len(params["arms"])))
    R=np.zeros((params["maxEpisodes"]))
    # actions=np.zeros((params["maxEpisodes"]))
    env.reset()
    epsilon=params["epsilon"]
    while e < params["maxEpisodes"]-1 :
        if random.random() > epsilon:
            max_indices=np.where(Q==np.amax(Q))
            harvest = random.choice(max_indices[0])
        else :
            harvest= random.choice(np.arange(1,len(Q)+1))
        print(harvest)
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
        
        if type=='lin':
            epsilon=epsilon-params["decay_rate"]['lin']
        else :
            epsilon = epsilon*np.exp(-params["decay_rate"]['exp'])
            
        R[e]=r
        e = e+1
        Q_est[e] = Q
        env.reset()
    return R
print(decayingEpsilonGreedy(env,params,"exp"))