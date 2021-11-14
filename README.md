# Foraging_DRL_Project

This is the group project of the courrse CS698R **Deep Reinforcement Learning** which explores the performance of various RL agents on a patch foraging environment. <br>
The code implementation for **Multi Arm Bandit** is in the `MAB_agents.ipynb` notebook. The notebook implements the problem as a multi arm bandit problem where each arm represents the number of times each patch must be harvested before the agent moves on to the next patch.
The code implementation for **Control Agents** is in the `ControlAgents.ipynb` notebook. The notebook implements various control agents on the environment to obtain optimal results.
The code implementation for **Deep Learning Agents** is in the `deep_agents_foraging-v0.ipynb` and `deep_agents_foraging-v1.ipynb` notebook. The `deep_agents_foraging-v0.ipynb` notebook implements **DQN**, **DDQN** and **VPG** on the linear patch with constant initial reward. The `deep_agents_foraging-v1.ipynb` notebook implements **DQN**, **DDQN** and **VPG** on the linear patch with rich and poor patches which have different initial rewards.



The environment is in the dir `gym-env/gym_env/envs/.`

There are five Environments, each for:
- Foraging with constant initial reward for tabular method.
- Foraging with variable initial reward for tabular method.
- Foraging with constant initial reward for deep learning method with revised state space.
- Foraging with varible initial reward for deep learning method with revised state space.
- A demo environment with visual rendering
 

## To install the gym environment, run:
```
git clone git@github.com:neel-singhania/Foraging_DRL_Project
cd myenv
pip install -e .
```
