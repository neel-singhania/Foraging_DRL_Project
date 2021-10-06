import gym
from gym import spaces
import numpy as np
import random
from gym.utils import seeding


class ForagingRichPoorPatch(gym.Env):

    def __init__(self, interval_time=10, total_time=4) -> None:
        
        self.action_space = spaces.Discrete(2)  # Only two actions possible, leave (0), and harvest (1).

        # Define required variables.
        self.reset()
        self.seed()
        # the time needed to traverse between the patches.
        self.interval_time = interval_time
        # have to recheck this harvest(decision) time with mentors
        self.decision_times = [1, 2, 3, 4]
        self.total_time = total_time*60
    
    def reset(self) :
        '''What to do if we reset the environment.
        '''
        # In our case simply reset the current state back to start state.
        self.state = 0
        self.elapsed_time = 0
        # introduced concept of a rich and poor patch by varying the starting reward from 2 to 14
        self.patchStartReward=self.np_random.choice((np.arange(2,15)))
        done=False
        return self.state,done
    
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action: int):
        '''Defines what to do if an action is taken.

        Parameters
        ----------
        action : int
            Action to take.

        Returns
        -------
        Tuple[int, float, bool, None]
            A tuple containing the next state, reward obtained, whether terminal state has been reached, and None.
        '''
        # If we harvest(1) increase state's value by 1, and for leaving(0) we clip it to 0.
        decision_time = self.np_random.choice(self.decision_times,1,[0.3, 0.4, 0.2, 0.1])
        done=False
        reward = 0
        
        if action == 1:
            if self.elapsed_time + decision_time > self.total_time:
                done=True
            else:
                reward = self.patchStartReward - 0.5*self.state + self.np_random.normal(0,0.025,1)
                self.state += 1
                self.elapsed_time += decision_time
        else:
            if self.elapsed_time + self.interval_time > self.total_time:
                done=True
            else:
                # start patch reward takes in a random choice from 2 values:
                # 1. Is the base reward when we encounter a poor patch
                # 2. Is the base reward when we encounter a rich patch
                # the reward for the poor patch is a random integer from 2-7
                # the reward for the rich patch is a random integer from 10-14
                self.patchStartReward=self.np_random.choice([self.np_random.randint(2,8),self.np_random.randint(10,15)])
                # the state of the patch resets to original value
                self.state = 0
                self.elapsed_time += self.interval_time
        
        # Return the next state, reward, episode end signal and an information object which could contain anything. We
        
        return self.state, reward, done
    
    def render(self):
        pass
