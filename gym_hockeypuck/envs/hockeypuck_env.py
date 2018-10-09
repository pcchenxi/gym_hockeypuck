#!/usr/bin/env python3

import sys
import time
sys.path.append('/home/xi/workspace/catkin_rlsim/src/lwrsim')
from lwrsim.lwrsim import LWRSIM
import numpy as np
from gym import spaces
import gym
from gym import error, spaces, utils
from gym.utils import seeding

class HockeypuckEnv(gym.Env):
    high = np.inf*np.ones(7)
    low = -high
    action_space = spaces.Box(low=low, high=high, dtype=np.float32)
    high = np.inf*np.ones(27)
    low = -high    
    observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
#   metadata = {'render.modes': ['human']}
    def compute_reward(self, action, init_puck_pos, good_dir, data):
        done = False
        speed_reward = 0
        force_reward = np.sqrt(np.sum(action*action))
        puck_pos = np.array(data['puck_pos'])
        puck_pos2 = np.array(data['puck2_pos'])
        blade_pos = np.array(data['blade_pos'])
        # print(blade_pos)

        dist_diff = (puck_pos - init_puck_pos)[:2]
        dist_moved = np.sqrt(np.sum(dist_diff*dist_diff))     

        dist_diff2 = (puck_pos2 - self.init_puck2_pos)[:2]
        dist_moved2 = np.sqrt(np.sum(dist_diff2*dist_diff2))    

        diff_blade_puck = (blade_pos - puck_pos)
        dist_blade_puck = np.sqrt(np.sum(diff_blade_puck*diff_blade_puck))   

        # print(dist_blade_puck)

        # if dist_moved > 0.01:
        #     # speed_reward = dist_moved
        #     speed_reward = dist_diff[0]*good_dir[0] + dist_diff[1]*good_dir[1]
        #     done = True
        #     dist_blade_puck = 0
        #     self.touch_puch = True

        # if dist_moved2 > 0.01 and self.touch_puch == False:
        #     done = True 
        #     speed_reward = -1

        reward = [-dist_blade_puck-force_reward*0.01, speed_reward]

        return reward, done


    def convert_to_state(self, data):
        joint_pos = np.array(data['joint_pos'])
        joint_vel = np.array(data['joint_vel'])
        puck_pos = np.array(data['puck_pos'])
        puck2_pos = np.array(data['puck2_pos'])
        tool_pos = np.array(data['tool_pos'])

        # index = np.asarray(['joint_pos', 'joint_vel', 'puck_pos', 'puc2k_pos'])
        # state = data[index]
        state = np.append(joint_pos, joint_vel)
        state = np.append(state, puck_pos)
        state = np.append(state, puck2_pos)
        state = np.append(state, tool_pos)
        return state

    def joint_imp_control(self):
        self.sim.init_controller(mode='JOINT_IMP_CTRL')

        trajectory = np.loadtxt('/home/xi/workspace/catkin_rlsim/src/lwrsim/data/hockey_joint/desired_pose_0.dat')

        self.sim.reset_to_home_pos()
        self.sim.set_home_pos({'pos': list(trajectory[0][1:])})
        self.sim.set_puck_pos({'pos': [-0.48674, 0.77046, 0.01762]})
        self.sim.set_puck2_pos({'pos': [-0.33295, 0.85376, 0.01963]})

        data = self.sim.get_full_data()
        joint_pos = data['joint_pos']
        for point in trajectory:
            self.sim.move_to_jnt_pos({'pos': point[1:]})
            # print(point)
            data = self.sim.get_full_data()
            reward, _ = self.compute_reward(np.array([0,0,0,0,0,0,0]), self.init_puck_pos, self.good_dir, data)   


    def __init__(self, render = False):
        self.show = False
        self.first = True 

        # self.sim = LWRSIM({
        #     'env_type': 'hockeypuck',
        #     'render': render,
        # })
        # self.reset()

    def seed(self, render = True):
        self.show = True


    def reset(self):
        if self.first:
            self.sim = LWRSIM({
                'env_type': 'hockeypuck',
                'render': self.show,
            })
            self.first = False
        # self.reset()   


        self.sim.init_controller(mode='JOINT_IMP_CTRL')
        self.sim.reset_to_home_pos()
        # self.sim.set_home_pos({'pos': [-0.889875, 0.751139, 0.0515551, -0.977962, 0.227244, -0.192762, 2.75731]})

        puck_pos1 = np.asarray([np.random.uniform(-1, 1) for i in range(2)])*0.3
        puck_pos2 = np.asarray([np.random.uniform(-1, 1) for i in range(2)])*0.3

        self.sim.set_puck_pos({'pos': [-0.2+puck_pos1[0], 1+puck_pos1[1], 0.02762]})
        self.sim.set_puck2_pos({'pos': [-0.2+puck_pos2[0], 1+puck_pos2[1], 0.02963]})

        data = self.sim.get_full_data()
        state = self.convert_to_state(data)

        self.init_puck_pos = np.array(data['puck_pos'])
        self.init_puck2_pos = np.array(data['puck2_pos'])

        puck2_dir = (self.init_puck2_pos - self.init_puck_pos)[:2]
        dist = np.sqrt(np.sum(puck2_dir*puck2_dir))

        self.good_dir = puck2_dir/dist
        self.touch_puch = False

        return state


    def step(self, action):
        # print(action)
        # action = np.clip(action, -1, 1)
        data = self.sim.get_full_data()

        joint_pos = data['joint_pos']
        force_tool = data['tool_force']
        # print(force_tool)
        # random_pose = [np.random.uniform(-1, 1) for i in range(7)]
        self.sim.move_to_jnt_pos({'pos': joint_pos + np.asarray(action)*0.1})

        data = self.sim.get_full_data()
        state = self.convert_to_state(data)

        reward, done = self.compute_reward(action, self.init_puck_pos, self.good_dir, data)    

        if done:
            joint_pos = data['joint_pos']
            for _ in range(20):
                self.sim.move_to_jnt_pos({'pos': joint_pos})

        reward, _ = self.compute_reward(action, self.init_puck_pos, self.good_dir, data)   
        return state, reward, done, {}

        # return state, reward, done, {} 


    def render(self, mode='human', close=False):
        a = 0

    def test_joint(self):
        # for pos in range(-360, 360, 10):
        self.sim.move_to_jnt_pos({'pos': [0, 0, 0, 0, 0, 0, 100]})
        data = self.sim.get_full_data()
        joint_pos = data['joint_pos']    
        print(joint_pos)        
        time.sleep(1)

        self.sim.move_to_jnt_pos({'pos': [0, 0, 0, 0, 0, 0, -365]})
        data = self.sim.get_full_data()
        joint_pos = data['joint_pos']    
        print(joint_pos)        
        time.sleep(1)

