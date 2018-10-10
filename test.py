import gym
import gym_hockeypuck
import numpy as np

env = gym.make('hockeypuck-v0')
env.seed()
print (env.action_space)
# for _ in range(30):
#     env.reset()
#     for _ in range(300):
#         # sim.move_to_jnt_pos({'pos': point[1:]})
#         random_pose = np.array([np.random.uniform(-1, 1) for i in range(7)])
#         state, r, done, _ = env.step(random_pose)
#         if done:
#             for _ in range(20):
#                 action = [0, 0, 0, 0, 0, 0, 0]
#                 _, r, _, _ = env.step(random_pose)
#             print(r, done)
#             break
    
