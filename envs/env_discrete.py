# -*- coding: utf-8 -*-
"""
离散动作环境封装：将 EnvCore 包装为符合 Runner 期望的多智能体环境。
- 使用 gym.spaces.MultiDiscrete（具有 .nvec），动作维度来自 EnvCore.action_dim（例如 [3, 3]）。
- 暴露 configure / load_channel / save_channel / seed，便于 UI 注入参数与固定信道复用。
- 与 DummyVecEnv 对齐：step/reset 返回的 obs/rews/dones 在此转为 np.stack。
"""
from __future__ import annotations

import numpy as np
from gym import spaces

from envs.env_core import EnvCore


class DiscreteActionEnv(object):
    """
    对离散动作环境的封装（多智能体）。
    每个 agent 的动作采用 MultiDiscrete(one-hot 展开的子动作)，比如 [3,3] 表示：
      - 前 3 维：{JT, CB, NonComp}
      - 后 3 维：{cluster_size = 3, 4, 5}（受 cluster_max 限制，EnvCore 里做裁剪）
    观测维度：1 + cluster_max（[SINR] + [用户最强 cluster_max 个 AP 的聚合负载]）。
    """

    def __init__(self):
        self.env = EnvCore()
        # 依据底层 EnvCore 构建 Gym spaces
        self._refresh_and_build_spaces()

    # -------------------------------------------------------------------------
    # 构建/刷新空间
    # -------------------------------------------------------------------------
    def _refresh_and_build_spaces(self):
        # 从 EnvCore 读取当前规模/维度
        self.num_agent = int(self.env.agent_num)
        self.signal_obs_dim = int(self.env.obs_dim)
        # e.g. [3,3]，会传给 gym.spaces.MultiDiscrete
        self.signal_action_dim = np.asarray(self.env.action_dim, dtype=int)

        # 清空并重建
        self.action_space = []
        self.observation_space = []
        self.share_observation_space = []

        # 共享观测拼接后的维度（= num_agent * obs_dim）
        share_obs_dim = 0

        for _ in range(self.num_agent):
            # 使用 Gym 官方 MultiDiscrete（注意，这里传的是 nvec）
            u_action_space = spaces.MultiDiscrete(self.signal_action_dim)
            self.action_space.append(u_action_space)

            # 每个 agent 的观测空间
            self.observation_space.append(
                spaces.Box(
                    low=-np.inf,
                    high=+np.inf,
                    shape=(self.signal_obs_dim,),
                    dtype=np.float32,
                )
            )
            share_obs_dim += self.signal_obs_dim

        # 所有 agent 共享观测（拼接各自 obs）
        self.share_observation_space = [
            spaces.Box(
                low=-np.inf,
                high=+np.inf,
                shape=(share_obs_dim,),
                dtype=np.float32,
            )
            for _ in range(self.num_agent)
        ]

    # -------------------------------------------------------------------------
    # UI/命令行参数注入（透传给 EnvCore），并根据变化重建 space
    # -------------------------------------------------------------------------
    def configure(self, all_args):
        """
        将 UI/命令行参数传递给 EnvCore：
        可读属性包括：num_aps, num_agents, map_size, ap_min_distance,
                    antennas_per_ap, channel_file, seed 等。
        """
        kwargs = dict(
            num_aps=getattr(all_args, "num_aps", None),
            num_agents=getattr(all_args, "num_agents", None),
            map_size=getattr(all_args, "map_size", None),
            ap_min_distance=getattr(all_args, "ap_min_distance", None),
            antennas_per_ap=getattr(all_args, "antennas_per_ap", None),
            channel_file=getattr(all_args, "channel_file", None),
            seed=getattr(all_args, "seed", None),
        )
        # 去掉 None
        kwargs = {k: v for k, v in kwargs.items() if v is not None}
        if kwargs:
            self.env.configure(**kwargs)
            # 规模/维度可能变化，重建空间
            self._refresh_and_build_spaces()

    # -------------------------------------------------------------------------
    # 与 DummyVecEnv 对齐的交互接口
    # -------------------------------------------------------------------------
    def step(self, actions):
        """
        输入：
            actions: [num_agent, one-hot 串] 由 Runner 组装（MultiDiscrete 的每个子动作已经选择好）
        返回（与 DummyVecEnv 对齐）：
            obs:   np.stack([obs_i])  -> [num_agent, obs_dim]
            rews:  np.stack([ [r_i] ])-> [num_agent, 1]
            dones: np.stack([done_i])-> [num_agent]
            infos: List[dict]         -> 长度 = num_agent
        """
        obs, rews, dones, infos = self.env.step(actions)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        obs = self.env.reset()
        return np.stack(obs)

    def close(self):
        # 底层目前无资源，需要时可以在 EnvCore 增加 close()
        if hasattr(self.env, "close"):
            self.env.close()

    def render(self, mode="rgb_array"):
        # 演示/可视化可在 EnvCore 实现；此处按需透传
        if hasattr(self.env, "render"):
            return self.env.render(mode=mode)
        return None

    def seed(self, seed):
        if hasattr(self.env, "seed"):
            self.env.seed(seed)

    # -------------------------------------------------------------------------
    # 固定信道场景的加载/保存（供 UI 与评估共用）
    # -------------------------------------------------------------------------
    def load_channel(self, path_or_dict):
        """
        加载固定信道：npz/json 或 dict（见 utils.channel_api.generate_channel 的格式）
        """
        ok = self.env.load_channel(path_or_dict)
        if ok:
            # 载入固定场景后，观测/动作维度不变，但为了稳妥，刷新一次空间
            self._refresh_and_build_spaces()
        return ok

    def save_channel(self, path):
        """
        保存当前场景为 npz/json（具体落盘由 EnvCore/Channel API 完成）
        """
        return self.env.save_channel(path)


# ======================================================================
# 重要提醒：
# 若你的旧文件底部还定义了一个自定义的 `class MultiDiscrete`，
# 请务必删除/注释掉那个类，避免覆盖 Gym 的 `spaces.MultiDiscrete`。
# 目前 Runner 的 MultiDiscrete one-hot 展开逻辑已经兼容 Gym 的 `.nvec`。
# ======================================================================
