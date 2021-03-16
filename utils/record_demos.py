import os
import warnings
from typing import Dict

import cv2  # pytype:disable=import-error
import numpy as np
from gym import spaces

from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.vec_env import VecEnv, VecFrameStack
from stable_baselines.common.base_class import _UnvecWrapper
from pathlib import Path
from gym_minigrid.wrappers import HumanFOVWrapper
import gym_minigrid
import gym

from os import listdir
from os.path import isfile, join
import utils.action_parser as action_parser
from utils.action_parser import json_to_action


# RESOURCES_DIR = (Path(__file__).parent / './human_subjects_data').resolve()


def generate_expert_traj(test_set_ratio):
    """
    Record expert trajectories for Falcon training.

    .. note::

        only Box and Discrete spaces are supported for now.
    """
    RESOURCES_DIR = (Path(__file__).parent / '../human_subjects_data').resolve()
    files = [f for f in listdir(RESOURCES_DIR) if isfile(join(RESOURCES_DIR, f))]

    e_y_demos = {}
    e_o_demos = {}
    m_y_demos = {}
    m_o_demos = {}
    d_y_demos = {}
    d_o_demos = {}

    e_y_num = 0
    e_o_num = 0
    m_y_num = 0
    m_o_num = 0
    d_y_num = 0
    d_o_num = 0

    for file_name in files:
        if 'HSRData_TrialMessages_CondBtwn' in file_name:
            print('processing ' + file_name)
            actions, invalid_data_found, strategy, episode_reward = json_to_action(file_name)
            if not invalid_data_found:
                if 'FalconEasy' in file_name:
                    if strategy == 'yellow':
                        e_y_num += 1
                        e_y_demos = collect_demos(e_y_demos, actions, 'easy')
                    else:
                        e_o_num += 1
                        e_o_demos = collect_demos(e_o_demos, actions, 'easy')
                elif 'FalconMed' in file_name:
                    if strategy == 'yellow':
                        m_y_num += 1
                        m_y_demos = collect_demos(m_y_demos, actions, 'medium')
                    else:
                        m_o_num += 1
                        m_o_demos = collect_demos(m_o_demos, actions, 'medium')
                elif 'FalconHard' in file_name:
                    if strategy == 'yellow':
                        d_y_num += 1
                        d_y_demos = collect_demos(d_y_demos, actions, 'difficult')
                    else:
                        d_o_num += 1
                        d_o_demos = collect_demos(d_o_demos, actions, 'difficult')
            else:
                print('failed file: ' + file_name)
                action_parser.invalid_data_found = False

    gen_training_and_test_sets(e_y_demos, 'easy', 'yellow', test_set_ratio, e_y_num)
    gen_training_and_test_sets(e_o_demos, 'easy', 'opportunistic', test_set_ratio, e_o_num)
    gen_training_and_test_sets(m_y_demos, 'medium', 'yellow', test_set_ratio, m_y_num)
    gen_training_and_test_sets(m_o_demos, 'medium', 'opportunistic', test_set_ratio, m_o_num)
    gen_training_and_test_sets(d_y_demos, 'difficult', 'yellow', test_set_ratio, d_y_num)
    gen_training_and_test_sets(d_o_demos, 'difficult', 'opportunistic', test_set_ratio, d_o_num)

    # e_y_demos = wrap_data(e_y_demos)
    # e_o_demos = wrap_data(e_o_demos, 'easy')
    # m_y_demos = wrap_data(m_y_demos, 'medium')
    # m_o_demos = wrap_data(m_o_demos, 'medium')
    # d_y_demos = wrap_data(d_y_demos, 'difficult')
    # d_o_demos = wrap_data(d_o_demos, 'difficult')
    #
    # np.savez('expert_data/falcon_easy_yellow', **e_y_demos)
    # np.savez('expert_data/falcon_easy_opportunistic', **e_o_demos)
    # np.savez('expert_data/falcon_medium_yellow', **m_y_demos)
    # np.savez('expert_data/falcon_medium_opportunistic', **m_o_demos)
    # np.savez('expert_data/falcon_difficult_yellow', **d_y_demos)
    # np.savez('expert_data/falcon_difficult_opportunistic', **d_o_demos)


def collect_demos(demonstrations, expert_actions, difficulty):
    # save_path = 'expert_data'
    env = gym.make('MiniGrid-MinimapForFalcon-v0', difficulty=difficulty)
    env = HumanFOVWrapper(env)

    if demonstrations == {}:
        actions = []
        observations = []
        rewards = []
        episode_returns = []
        episode_starts = []
    else:
        actions = demonstrations['actions']
        observations = demonstrations['obs']
        rewards = demonstrations['rewards']
        episode_returns = demonstrations['episode_returns']
        episode_starts = demonstrations['episode_starts']

    obs = env.reset()
    episode_starts.append(True)
    reward_sum = 0.0

    for action in expert_actions:
        observations.append(obs)
        obs, reward, done, _ = env.step(action)

        actions.append(action)
        rewards.append(reward)
        episode_starts.append(done)
        reward_sum += reward

        if done:
            episode_starts.pop(-1)
            episode_returns.append(reward_sum)
            break

    assert len(observations) == len(actions)

    numpy_dict = {
        'actions': actions,
        'obs': observations,
        'rewards': rewards,
        'episode_returns': episode_returns,
        'episode_starts': episode_starts
    }  # type: Dict[str, list]

    env.close()

    return numpy_dict


def wrap_data(demonstrations):
    env = gym.make('MiniGrid-MinimapForFalcon-v0')
    env = HumanFOVWrapper(env)

    if demonstrations == {}:
        actions = []
        observations = []
        rewards = []
        episode_returns = []
        episode_starts = []
    else:
        actions = demonstrations['actions']
        observations = demonstrations['obs']
        rewards = demonstrations['rewards']
        episode_returns = demonstrations['episode_returns']
        episode_starts = demonstrations['episode_starts']

    if isinstance(env.observation_space, spaces.Box):
        observations = np.concatenate(observations).reshape((-1,) + env.observation_space.shape)
    elif isinstance(env.observation_space, spaces.Discrete):
        observations = np.array(observations).reshape((-1, 1))

    if isinstance(env.action_space, spaces.Box):
        actions = np.concatenate(actions).reshape((-1,) + env.action_space.shape)
    elif isinstance(env.action_space, spaces.Discrete):
        actions = np.array(actions).reshape((-1, 1))

    rewards = np.array(rewards)
    # episode_starts = np.array(episode_starts[:-1])
    episode_starts = np.array(episode_starts)

    assert len(observations) == len(actions)

    numpy_dict = {
        'actions': actions,
        'obs': observations,
        'rewards': rewards,
        'episode_returns': np.array(episode_returns),
        'episode_starts': episode_starts
    }  # type: Dict[str, np.ndarray]

    return numpy_dict


# According to the number of demonstrations and the test set ratio, we seperate the demonstrations into the training and test sets
def gen_training_and_test_sets(demonstrations, difficulty, strategy, test_set_ratio, data_num):
    training_dict = {
        'actions': [],
        'obs': [],
        'rewards': [],
        'episode_returns': [],
        'episode_starts': []
    }  # type: Dict[str, list]

    test_dict = {
        'actions': [],
        'obs': [],
        'rewards': [],
        'episode_returns': [],
        'episode_starts': []
    }  # type: Dict[str, list]

    training_file = 'falcon_' + difficulty + '_' + strategy + '_training'
    test_file = 'falcon_' + difficulty + '_' + strategy + '_test'
    demos_len = data_num

    if demos_len < 2:
        training_set = training_dict
        test_set = test_dict
        np.savez('expert_data/' + training_file, **training_set)
        np.savez('expert_data/' + test_file, **test_set)
        return

    test_set_num = int(max(np.ceil(demos_len * test_set_ratio), 1))
    test_idx = np.sort(np.random.choice(range(demos_len), test_set_num, replace=False)).tolist()
    training_idx = [i for i in range(data_num) if i not in test_idx]
    episode_starts_idx = [i for i, e in enumerate(demonstrations['episode_starts']) if e]

    for ti in test_idx:
        start_idx = episode_starts_idx[ti]
        if ti == len(episode_starts_idx) - 1:
            end_idx = len(demonstrations['actions'])
        else:
            end_idx = episode_starts_idx[ti + 1]
        test_dict['actions'] += demonstrations['actions'][start_idx: end_idx]
        test_dict['obs'] += demonstrations['obs'][start_idx: end_idx]
        test_dict['rewards'] += demonstrations['rewards'][start_idx: end_idx]
        test_dict['episode_returns'].append(demonstrations['episode_returns'][ti])
        test_dict['episode_starts'] += demonstrations['episode_starts'][start_idx: end_idx]

    for ti in training_idx:
        start_idx = episode_starts_idx[ti]
        if ti == len(episode_starts_idx) - 1:
            end_idx = len(demonstrations['actions'])
        else:
            end_idx = episode_starts_idx[ti + 1]
        training_dict['actions'] += demonstrations['actions'][start_idx: end_idx]
        training_dict['obs'] += demonstrations['obs'][start_idx: end_idx]
        training_dict['rewards'] += demonstrations['rewards'][start_idx: end_idx]
        training_dict['episode_returns'].append(demonstrations['episode_returns'][ti])
        training_dict['episode_starts'] += demonstrations['episode_starts'][start_idx: end_idx]

    training_set = wrap_data(training_dict)
    test_set = wrap_data(test_dict)

    np.savez('expert_data/' + training_file, **training_set)
    np.savez('expert_data/' + test_file, **test_set)
