from gym import Env
from gym.spaces import  Discrete
from gym.utils import seeding
import numpy as np
import random

class env(Env):
    
    def __init__(self,TT,maxTime):
        # action space harvest =0,leave=1
        self.action_space = Discrete(2)
        # setting time to 0
        self.time=0
        self.n = 0
        self.travelTime=TT #travel time is a function of the environment
        self.maxTime=maxTime
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action, patch_quality):
        info={}
        # harvest time includes decision time and harvest time
        # decision time is 1 +/- 0.6 sec and harvest takes no time.
        self.harvestTime = random.uniform(0.4,1.6) 
        if action == 0:
            if self.time+self.harvestTime<=self.maxTime:
                self.time+=self.harvestTime
                self.n+=1
                if patch_quality=="rich":
                    # for rich patch the maximum reward is a uniform random variable which takes value random value from [10,11,12,13,14] 
                    reward=random.randint(10,15)-0.5*self.n+self.np_random.normal(0,0.025)
                else:
                    # for poor patch the maximum reward is a uniform random variable which takes value random value from [2,3,4,5,6]
                    reward=random.randint(2,7)-0.5*self.n+self.np_random.normal(0,0.025)
                done=False
            else:
                reward=0
                done=True
                
        else :
            if self.time+self.travelTime<self.maxTime:
                self.n=0  
                self.time+=self.travelTime
                reward=0
                done=False
            else:
                reward=0
                done=True
        return self.n,reward,done,info
    
    def reset(self):
        self.time=0
        self.n=0
        done=False
        return self.n,done

    def render(self):
        pass

