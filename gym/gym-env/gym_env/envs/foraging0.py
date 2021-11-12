import gym
from gym import spaces
import numpy as np
import random
import torch


class Foraging0(gym.Env):

    def __init__(self, interval_time=10, total_time=4) -> None:
        '''Constructor for our environment. Should take any relevant parameters as arguments.

        Parameters
        ----------
        n_states : int
            The number of states for the bandit walk environment.
        '''
        super(Foraging0, self).__init__()
        self.action_space = spaces.Discrete(2)  # Only two actions possible, leave (0), and harvest (1).

        # Define required variables.
        self.reset()
        self.interval_time = interval_time
        self.decision_times = [1, 0.8, 1.4, 0.6]
        self.total_time = total_time*60

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
        decision_time = np.random.choice(self.decision_times,1,[0.3, 0.4, 0.2, 0.1])
        done=False
        reward = 0
        
        if action == 1:
            if self.elapsed_time + decision_time > self.total_time:
                done=True
            else:
                reward = 7 - 0.5*self.count + np.random.normal(0,0.025,1)
                if reward<0: reward = 0
                self.count += 1
                self.elapsed_time += decision_time
        else:
            if self.elapsed_time + self.interval_time > self.total_time:
                done=True
            else:
                self.count = 0
                self.elapsed_time += self.interval_time
        self.state = self.state[1:5]
        if isinstance(reward, np.ndarray):
            reward = reward[0]
        self.state.append(reward)
        self.state.append(action)
        # Return the next state, reward, episode end signal and an information object which could contain anything. We
        # don't have any additional info to return so we return None.
        return self.state, reward, done, None

    def reset(self) -> None:
        '''What to do if we reset the environment.
        '''
        # In our case simply reset the current state back to start state.
        self.state = [0, 0, 0, 0, 0, 0]
        self.count = 0
        self.elapsed_time = 0
        self.reward_mean = 7
        return self.state, False
    
    def seed(self,seed=None) -> None:
        np.random.seed(seed)
        random.seed(seed)