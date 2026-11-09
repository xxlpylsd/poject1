from __future__ import annotations
import numpy as np
from gym import spaces
from envs.env_core import EnvCore

class DiscreteActionEnv(object):

    def __init__(self):
        self.env = EnvCore()
        self._refresh_and_build_spaces()

    def _refresh_and_build_spaces(self):
        self.num_agent = int(self.env.agent_num)
        self.signal_obs_dim = int(self.env.obs_dim)
        self.signal_action_dim = np.asarray(self.env.action_dim, dtype=int)
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []
        share_obs_dim = 0
        for _ in range(self.num_agent):
            u_action_space = spaces.MultiDiscrete(self.signal_action_dim)
            self.action_space.append(u_action_space)
            self.observation_space.append(spaces.Box(low=-np.inf, high=+np.inf, shape=(self.signal_obs_dim,), dtype=np.float32))
            share_obs_dim += self.signal_obs_dim
        self.share_observation_space = [spaces.Box(low=-np.inf, high=+np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in range(self.num_agent)]

    def configure(self, all_args):
        kwargs = dict(num_aps=getattr(all_args, 'num_aps', None), num_agents=getattr(all_args, 'num_agents', None), map_size=getattr(all_args, 'map_size', None), ap_min_distance=getattr(all_args, 'ap_min_distance', None), antennas_per_ap=getattr(all_args, 'antennas_per_ap', None), channel_file=getattr(all_args, 'channel_file', None), seed=getattr(all_args, 'seed', None))
        kwargs = {k: v for (k, v) in kwargs.items() if v is not None}
        if kwargs:
            self.env.configure(**kwargs)
            self._refresh_and_build_spaces()

    def step(self, actions):
        (obs, rews, dones, infos) = self.env.step(actions)
        return (np.stack(obs), np.stack(rews), np.stack(dones), infos)

    def reset(self):
        obs = self.env.reset()
        return np.stack(obs)

    def close(self):
        if hasattr(self.env, 'close'):
            self.env.close()

    def render(self, mode='rgb_array'):
        if hasattr(self.env, 'render'):
            return self.env.render(mode=mode)
        return None

    def seed(self, seed):
        if hasattr(self.env, 'seed'):
            self.env.seed(seed)

    def load_channel(self, path_or_dict):
        ok = self.env.load_channel(path_or_dict)
        if ok:
            self._refresh_and_build_spaces()
        return ok

    def save_channel(self, path):
        return self.env.save_channel(path)
