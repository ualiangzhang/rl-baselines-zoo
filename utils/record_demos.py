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


def generate_expert_traj():
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

    e_y_demos = wrap_data(e_y_demos, 'easy')
    e_o_demos = wrap_data(e_o_demos, 'easy')
    m_y_demos = wrap_data(m_y_demos, 'medium')
    m_o_demos = wrap_data(m_o_demos, 'medium')
    d_y_demos = wrap_data(d_y_demos, 'difficult')
    d_o_demos = wrap_data(d_o_demos, 'difficult')

    np.savez('expert_data/falcon_easy_yellow', **e_y_demos)
    np.savez('expert_data/falcon_easy_opportunistic', **e_o_demos)
    np.savez('expert_data/falcon_medium_yellow', **m_y_demos)
    np.savez('expert_data/falcon_medium_opportunistic', **m_o_demos)
    np.savez('expert_data/falcon_difficult_yellow', **d_y_demos)
    np.savez('expert_data/falcon_difficult_opportunistic', **d_o_demos)


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

        # if isinstance(model, BaseRLModel):
        #     action, state = model.predict(obs, state=state, mask=mask)
        # else:
        #     action = model(obs)

        obs, reward, done, _ = env.step(action)

        actions.append(action)
        rewards.append(reward)
        episode_starts.append(done)
        reward_sum += reward

        if done:
            # obs = env.reset()
            episode_returns.append(reward_sum)
            # reward_sum = 0.0
            break

    # if isinstance(env.observation_space, spaces.Box):
    #     observations = np.concatenate(observations).reshape((-1,) + env.observation_space.shape)
    # elif isinstance(env.observation_space, spaces.Discrete):
    #     observations = np.array(observations).reshape((-1, 1))
    #
    # if isinstance(env.action_space, spaces.Box):
    #     actions = np.concatenate(actions).reshape((-1,) + env.action_space.shape)
    # elif isinstance(env.action_space, spaces.Discrete):
    #     actions = np.array(actions).reshape((-1, 1))
    #
    # rewards = np.array(rewards)
    # episode_starts = np.array(episode_starts[:-1])

    assert len(observations) == len(actions)

    numpy_dict = {
        'actions': actions,
        'obs': observations,
        'rewards': rewards,
        'episode_returns': episode_returns,
        'episode_starts': episode_starts
    }  # type: Dict[str, list]

    # for key, val in numpy_dict.items():
    #     print(key, val.shape)

    # if save_path is not None:
    #     np.savez(save_path, **numpy_dict)

    env.close()

    return numpy_dict


def wrap_data(demonstrations, difficulty):
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

    if isinstance(env.observation_space, spaces.Box):
        observations = np.concatenate(observations).reshape((-1,) + env.observation_space.shape)
    elif isinstance(env.observation_space, spaces.Discrete):
        observations = np.array(observations).reshape((-1, 1))

    if isinstance(env.action_space, spaces.Box):
        actions = np.concatenate(actions).reshape((-1,) + env.action_space.shape)
    elif isinstance(env.action_space, spaces.Discrete):
        actions = np.array(actions).reshape((-1, 1))

    rewards = np.array(rewards)
    episode_starts = np.array(episode_starts[:-1])

    assert len(observations) == len(actions)

    numpy_dict = {
        'actions': actions,
        'obs': observations,
        'rewards': rewards,
        'episode_returns': np.array(episode_returns),
        'episode_starts': episode_starts
    }  # type: Dict[str, np.ndarray]

    return numpy_dict

# generate_expert_traj()
