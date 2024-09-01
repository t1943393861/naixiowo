import argparse
import time

import cv2.cv2
import numpy as np
from copy import deepcopy
from skimage.transform import resize
import torch
from xuance.common import get_configs, recursive_dict_update
from xuance.environment import make_envs
from xuance.torch.utils.operations import set_seed
from xuance.torch.agents import DQN_Agent
import gym
from matplotlib import pyplot as plt
import random
from IPython import display
from collections import Counter

# 打印游戏
def show():
    plt.imshow(env.render())
    plt.show()


# 定义环境
class MyWrapper(gym.Wrapper):
    def __init__(self):
        env = gym.make('ALE/Breakout-v5', apply_api_compatibility=True, render_mode='rgb_array')
        super().__init__(env)
        self.env = env
        self.step_n = 0

    def reset(self):
        state = self.env.reset()
        self.step_n = 0
        return state

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self.step_n += 1
        if self.step_n >= 50000:
            done = True
        return state, reward, done, truncated, info


def parse_args():
    parser = argparse.ArgumentParser('Exaple for XuanCe: DQN for atari')
    parser.add_argument('--env-id', type=str, default='ALE/Breakout-v5')
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--benchmark", type=int, default=1)

    return parser.parse_args()


def get_action(state):
    new_channel = np.zeros((84, 84, 1))
    state = cv2.cv2.resize(state, (84, 84))
    state = np.concatenate([state, new_channel], axis=-1)
    state = np.expand_dims(state, axis=0)
    values = Agent.action(state)['actions'].tolist()
    # action = random.choice(values)
    return values


def test(play):
    # 初始化游戏
    state = env.reset()[0]
    # 记录反馈和
    reward_sum = 0
    # 玩到游戏结束为止
    over = False
    while not over:
        actions = get_action(state)
        for action in actions:
            new_actions = get_action(state)
            actions.extend(new_actions)
            next_state, reward, over, _1, _2 = env.step(action)
            if over:
                break
            reward_sum += reward
            state = next_state
            plt.imshow(env.render())
            plt.show(block=False)
            plt.pause(0.01)  # 暂停
            plt.clf()
    return reward_sum


path = '.\\models\\dqn'
if __name__ == "__main__":
    parser = parse_args()
    configs_dict = get_configs(file_dir='.\\dqn_configs\\dqn_atari_config.yaml')
    configs_dict = recursive_dict_update(configs_dict, parser.__dict__)
    configs = argparse.Namespace(**configs_dict)

    set_seed(configs.seed)
    envs = make_envs(configs)
    Agent = DQN_Agent(config=configs, envs=envs)
    Agent.load_model(path=path)
    env = MyWrapper()
    state = env.reset()[0]
    reward_sum = test(play=True)
    print(f'reward_sum=', reward_sum)