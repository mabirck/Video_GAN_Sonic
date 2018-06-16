"""
Environments and wrappers for Sonic training.
"""

import gym
import cv2
import numpy as np
from gym import spaces

from baselines.common.atari_wrappers import FrameStack
from utils import WrapPyTorch, getListOfGames

try:
    from retro_contest.local import make
except ImportError:
    print("FAILED TO IMPORT retro_contest.local")


def make_env(env_id, stack=True, scale_rew=True):
    """
    Create an environment with some standard wrappers.
    """

    game, state = env_id.split(',')
    print("MAKING",game, "STATE", state)

    env = make(game=game, state=state)
    env = SonicDiscretizer(env)

    if scale_rew:
        env = RewardScaler(env)
    env = WarpFrame(env, (32, 32))
    if stack:
        env = FrameStack(env, 4)

    return WrapPyTorch(env)


class SonicDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the Sonic game.
    """

    def __init__(self, env):
        super(SonicDiscretizer, self).__init__(env)
        buttons = ["B", "A", "MODE", "START", "UP", "DOWN", "LEFT", "RIGHT", "C", "Y", "X", "Z"]
        actions = [['LEFT'], ['RIGHT'], ['LEFT', 'DOWN'], ['RIGHT', 'DOWN'], ['DOWN'],
                   ['DOWN', 'B'], ['B']]
        self._actions = []
        for action in actions:
            arr = np.array([False] * 12)
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a): # pylint: disable=W0221
        return self._actions[a].copy()


class RewardScaler(gym.RewardWrapper):
    """
    Bring rewards to a reasonable scale for PPO.
    This is incredibly important and effects performance
    drastically.
    """

    def reward(self, reward):
        return reward * 0.01


class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """

    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs): # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action): # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env, size=(84, 84)):
        """Warp frames to 84x84 as done in the Nature paper and later work."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = size[0]
        self.height = size[1]
        self.observation_space = spaces.Box(low=0, high=255,
            shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]

def make_env_train(env_id, seed, rank, log_dir):
    def _thunk():
        env = make_env(env_id=env_id)

        env.seed(seed+rank)
        # if log_dir is not None:
        #     env = bench.Monitor(env, os.path.join(log_dir, str(rank)))
        return env
    return _thunk
