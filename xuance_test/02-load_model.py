import argparse
import time

import numpy as np
from copy import deepcopy

import torch
from xuance.common import get_configs, recursive_dict_update
from xuance.environment import make_envs
from xuance.torch.utils.operations import set_seed
from xuance.torch.agents import DQN_Agent
import gym
from matplotlib import pyplot as plt
import random
from IPython import display


# 打印游戏
def show():
    plt.imshow(env.render(mode='rgb_array'))
    plt.show()


# 定义环境
class MyWrapper(gym.Wrapper):
    def __init__(self):
        env = gym.make('ALE/Breakout-v5')
        super().__init__(env)
        self.env = env
        self.step_n = 0

    def reset(self):
        state = self.env.reset()
        self.step_n = 0
        return state

    def step(self, action):
        state, reward, terminated, info = self.env.step(action)
        done = terminated
        self.step_n += 1
        if self.step_n >= 5000:
            done = True
        return state, reward, done, info


def parse_args():
    parser = argparse.ArgumentParser('Exaple for XuanCe: DQN for atari')
    parser.add_argument('--env-id', type=str, default='ALE/Breakout-v5')
    parser.add_argument("--test", type=int, default=0)
    parser.add_argument("--benchmark", type=int, default=1)

    return parser.parse_args()


def get_action(states):
    # (210, 160, 3, 4) -> (3, 210, 160, 4)
    states = np.transpose(states, (2, 0, 1, 3))
    actions = Agent.action(observations=states)['actions']
    return actions.tolist()


def state_trans(states_cache):
    states = np.array(states_cache)
    states = np.transpose(states, (1, 2, 3, 0))
    return states


def test(play):
    # 初始化游戏
    state = env.reset()
    states_cache = []
    # 把初始状态复制4份作为初始4帧的状态
    for _ in range(4):
        states_cache.append(state)
    states = np.stack(states_cache, axis=-1)
    # 记录反馈和
    reward_sum = 0
    # 玩到游戏结束为止
    over = False
    while not over:
        # 根据当前状态得到动作
        actions = get_action(states)
        states_cache.clear()
        for action in actions:
            next_state, reward, over, _ = env.step(action)
            reward_sum += reward
            states_cache.append(next_state)
            if over:
                return reward_sum
        states_cache.pop(0)
        if len(actions) < 4:
            for i in range(5 - len(actions)):
                states_cache.append(states_cache[-1])
        states = state_trans(states_cache)
        plt.imshow(env.render(mode='rgb_array'))
        plt.show(block=False)
        plt.pause(0.01)  # 暂停
        plt.clf()
    return reward_sum


path = 'E:\\Environment\\AI_Learning\\pytorch\\xuance_test\\models\\dqn'
if __name__ == "__main__":
    parser = parse_args()
    configs_dict = get_configs(file_dir='E:\\Environment\\AI_Learning\\pytorch\\xuance_test\\dqn_configs\\dqn_atari_config.yaml')
    configs_dict = recursive_dict_update(configs_dict, parser.__dict__)
    configs = argparse.Namespace(**configs_dict)

    set_seed(configs.seed)
    envs = make_envs(configs)
    Agent = DQN_Agent(config=configs, envs=envs)
    Agent.load_model(path=path)
    env = MyWrapper()
    state = env.reset()
    reward_sum = test(play=True)
    print(f'reward_sum=', reward_sum)
