from gym import Env
from gym.spaces import  Discrete
from gym.utils import seeding
import numpy as np
import random
from scipy.stats import bernoulli
from matplotlib import pyplot as plt

class env(Env):
    
    def __init__(self,HT,maxTime):
        # action space harvest =0,leave=1
        self.action_space = Discrete(2)
        # setting time to 0
        self.time=0
        self.harvestTime=HT
        self.maxTime=maxTime
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        info={}
        if action == 0:
            if self.time+self.harvestTime<=self.maxTime:
                self.time+=self.harvestTime
                reward=7-0.5*self.n+self.np_random.normal(0,0.025)
                self.n+=1
                done=False
            else:
                reward=0
                done=True
                
        else :
            travel_time=self.np_random.choice([3,10])
            if self.time+travel_time<self.maxTime:
                self.n=0  
                self.time+=travel_time
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

