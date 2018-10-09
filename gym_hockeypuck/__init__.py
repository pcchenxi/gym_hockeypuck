from gym.envs.registration import register

register(
    id='hockeypuck-v0',
    entry_point='gym_hockeypuck.envs:HockeypuckEnv',
    max_episode_steps=200,
)

