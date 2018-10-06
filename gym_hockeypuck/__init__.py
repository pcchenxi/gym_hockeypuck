from gym.envs.registration import register

register(
    id='hockeypuck-v0',
    entry_point='gym_hockeypuck.envs:HockeypuckEnv',
)

