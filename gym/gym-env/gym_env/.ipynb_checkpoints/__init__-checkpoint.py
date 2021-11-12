from gym.envs.registration import register

register(
    id='foraging-v0',  
    entry_point='gym_env.envs:Foraging0',
)
register(
    id='foraging-v1',  
    entry_point='gym_env.envs:Foraging1',
)
register(
    id='foraging-v2',  
    entry_point='gym_env.envs:Foraging2',
)
