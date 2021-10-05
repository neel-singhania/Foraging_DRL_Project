import gym
from gym import spaces
import numpy as np
import random


class forage_env(gym.Env):
    def __init__(self):
        self.reward = 0
        self.action_space = spaces.Discrete(2)
        self.n = 0
        self.time_limit = 240
        self.time = 0
        self.transtition_time = [3, 10]
        self.harvest_time = 1
        self.done = False

    def reset(self):
        self.reward = 0
        self.n = 0
        self.time = 0

    def step(self, action):

        self.harvest_time = random.uniform(0.4,1.6)
        if not self.done:
            if action == 0:
                temp = random.choice(self.transtition_time)
                if self.time + temp>=self.time_limit:
                    self.done = True
                else:
                    self.n = 1
                    self.time = self.time + temp
                    self.reward = 0

            if action == 1:
                if self.time + self.harvest_time>=self.time_limit:
                    self.done = True
                else:
                    self.reward = (7 - (0.5 * self.n) + np.random.normal(0, 0.025, 1)[0])
                    self.time += self.harvest_time
                    self.n += 1

            if self.time > self.time_limit:
                self.done = True
            
        return self.n, self.reward, self.done