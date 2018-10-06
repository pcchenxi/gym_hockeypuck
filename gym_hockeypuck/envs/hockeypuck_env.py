#!/usr/bin/env python3

import sys
sys.path.append('/home/xi/workspace/catkin_rlsim/src/lwrsim')
from lwrsim.lwrsim import LWRSIM
import numpy as np
import gym
from gym import error, spaces, utils
from gym.utils import seeding

class HockeypuckEnv(gym.Env):
#   metadata = {'render.modes': ['human']}

    def joint_imp_control(self, sim):
        sim.init_controller(mode='JOINT_IMP_CTRL')

        trajectory = np.loadtxt('/home/xi/workspace/catkin_rlsim/src/lwrsim/data/hockey_joint/desired_pose_0.dat')

        sim.reset_to_home_pos()
        sim.set_home_pos({'pos': list(trajectory[0][1:])})
        sim.set_puck_pos({'pos': [-0.48674, 0.77046, 0.02762]})
        sim.set_puck2_pos({'pos': [-0.33295, 0.85376, 0.02963]})

        # t = 0
        # while True:
        #     if t % 2000 == 0:
        #         random_pose = [np.random.uniform(-1, 1) for i in range(7)]
        #         sim.ctrl.target_pos = random_pose
        #         t = 0
        #     t += 1

        #     sim.step()

        data = sim.get_full_data()
        joint_pos = data['joint_pos']
        for point in trajectory:
            # sim.move_to_jnt_pos({'pos': point[1:]})
            random_pose = [np.random.uniform(-1, 1) for i in range(7)]
            sim.move_to_jnt_pos({'pos': joint_pos + np.asarray(random_pose)*0.01})
            # print(point)

            data = sim.get_full_data()
            joint_pos = data['joint_pos']
            # tool_pos = np.asarray(data['tool_pos'])[:3]
            # puck_pos = np.asarray(data['puck_pos'])[:3]

            # diff = tool_pos - puck_pos
            # dist = np.sqrt(np.sum(diff*diff))
            # print (dist)

    def __init__(self):
        self.sim = LWRSIM({
            'env_type': 'hockeypuck',
            'render': True,
        })

        # self.joint_imp_control(sim)
        self.reset()

    def reset(self):
        self.sim.init_controller(mode='JOINT_IMP_CTRL')
        self.sim.reset_to_home_pos()
        self.sim.set_home_pos({'pos': [-0.889875, 0.751139, 0.0515551, -0.977962, 0.227244, -0.192762, 2.75731]})

        puck_pos1 = np.asarray([np.random.uniform(-1, 1) for i in range(2)])*0.2
        puck_pos2 = np.asarray([np.random.uniform(-1, 1) for i in range(2)])*0.5

        self.sim.set_puck_pos({'pos': [-0.4+puck_pos1[0], 0.7+puck_pos1[1], 0.02762]})
        self.sim.set_puck2_pos({'pos': [-0.3+puck_pos1[0], 0.9+puck_pos1[1], 0.02963]})

        data = self.sim.get_full_data()
        print(data)
        joint_pos = data['joint_pos']
        for _ in range(500):
            # sim.move_to_jnt_pos({'pos': point[1:]})
            random_pose = [np.random.uniform(-1, 1) for i in range(7)]
            self.sim.move_to_jnt_pos({'pos': joint_pos + np.asarray(random_pose)*0.01})

            data = self.sim.get_full_data()
            joint_pos = data['joint_pos']


    # def step(self, action):

#   def step(self, action):
#     ...
#   def reset(self):
#     ...
#   def render(self, mode='human', close=False):
#     ...

# data_pre = []

# # def convert_state(data):
# #     tool_pos = np.asarray(data['tool_pos'])[:3]
# #     puck_pos = np.asarray(data['puck_pos'])[:3]    

# # def compute_reward(data)


# # def reset():
# #     state = 1

# #     return state

