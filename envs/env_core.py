from __future__ import annotations
import os, math
from typing import Dict, Any, Optional, Tuple, List
import numpy as np

def _maybe_import_channel_api():
    try:
        from utils.channel_api import ChannelConfig, generate_channel, save_channel, load_channel
        return (ChannelConfig, generate_channel, save_channel, load_channel)
    except Exception:
        return (None, None, None, None)

class EnvCore(object):
    DEFAULT_CLUSTER_MAX = 5
    DEFAULT_POWER_W = 0.2
    DEFAULT_BW_HZ = 20000000.0
    DEFAULT_BACKHAUL_RANGE = (400000000.0, 500000000.0)
    DEFAULT_MAP_SIZE = 1000.0

    def __init__(self):
        self.agent_num: int = 30
        self.ap_num: int = 50
        self.cluster_max: int = EnvCore.DEFAULT_CLUSTER_MAX
        self.obs_dim: int = 1 + self.cluster_max
        self.action_dim: List[int] = [3, 3]
        self.power_w: float = EnvCore.DEFAULT_POWER_W
        self.bandwidth_hz: float = EnvCore.DEFAULT_BW_HZ
        self.omega: float = 1.0
        self.map_size: float = EnvCore.DEFAULT_MAP_SIZE
        self.ap_min_distance: float = 0.0
        self.antennas_per_ap: int = 1
        self.backhaul_cap_range: Tuple[float, float] = EnvCore.DEFAULT_BACKHAUL_RANGE
        self.gains = {'jt': 2000.0, 'cb_main': 800.0, 'cb_others': 5, 'noncomp': 1000.0, 'base': 100.0}
        self.users_xy: Optional[np.ndarray] = None
        self.aps_xy: Optional[np.ndarray] = None
        self.backhaul_caps: Optional[np.ndarray] = None
        self.mu: Optional[np.ndarray] = None
        self.distance: Optional[np.ndarray] = None
        self._rng = np.random.default_rng(1)
        self._last_inst_rate = None
        self._last_interf_w = None
        self._reset_or_build_scenario()

    def seed(self, seed: Optional[int]):
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))

    def configure(self, num_aps: Optional[int]=None, num_agents: Optional[int]=None, map_size: Optional[float]=None, ap_min_distance: Optional[float]=None, antennas_per_ap: Optional[int]=None, channel_file: Optional[str]=None, seed: Optional[int]=None, backhaul_cap_range: Optional[Tuple[float, float]]=None, cluster_max: Optional[int]=None):
        if seed is not None:
            self.seed(seed)
        if num_aps is not None:
            self.ap_num = int(num_aps)
        if num_agents is not None:
            self.agent_num = int(num_agents)
        if map_size is not None:
            self.map_size = float(map_size)
        if ap_min_distance is not None:
            self.ap_min_distance = float(ap_min_distance)
        if antennas_per_ap is not None:
            self.antennas_per_ap = int(antennas_per_ap)
        if backhaul_cap_range is not None:
            self.backhaul_cap_range = backhaul_cap_range
        if cluster_max is not None:
            self.cluster_max = int(cluster_max)
            self.obs_dim = 1 + self.cluster_max
        if channel_file:
            self.load_channel(channel_file)
        else:
            self._reset_or_build_scenario()

    def load_channel(self, path_or_dict: Any) -> bool:
        (ChannelConfig, generate_channel, save_channel, load_channel) = _maybe_import_channel_api()
        try:
            if isinstance(path_or_dict, str):
                scn = load_channel(path_or_dict) if load_channel else {k: v for (k, v) in np.load(path_or_dict, allow_pickle=True).items()}
            elif isinstance(path_or_dict, dict):
                scn = path_or_dict
            else:
                return False
            self.agent_num = int(scn.get('M', self.agent_num))
            self.ap_num = int(scn.get('N', self.ap_num))
            self.map_size = float(scn.get('map_size', self.map_size))
            self.users_xy = np.array(scn['users'], dtype=float)
            self.aps_xy = np.array(scn['aps'], dtype=float)
            self.backhaul_caps = np.array(scn['caps'], dtype=float)
            self.mu = np.array(scn['mu'], dtype=float)
            self.distance = self._pairwise_dist(self.users_xy, self.aps_xy)
            return True
        except Exception:
            return False

    def save_channel(self, path: str) -> Optional[str]:
        if self.users_xy is None or self.aps_xy is None or self.mu is None:
            return None
        (ChannelConfig, generate_channel, save_channel, load_channel) = _maybe_import_channel_api()
        scn = {'M': self.agent_num, 'N': self.ap_num, 'map_size': self.map_size, 'ap_min_distance': self.ap_min_distance, 'users': self.users_xy, 'aps': self.aps_xy, 'caps': self.backhaul_caps, 'mu': self.mu}
        try:
            if save_channel:
                return save_channel(scn, path)
            os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
            np.savez_compressed(path, **scn)
            return path
        except Exception:
            return None

    def reset(self) -> List[np.ndarray]:
        return self._observe_initial()

    def step(self, actions: List[np.ndarray]):
        (mode_idx, cluster_k) = self._decode_actions(actions)
        (sinr, rate, inter_w, ap_load) = self._compute_rate(mode_idx, cluster_k)
        obs = self._observe_from_state(sinr, ap_load)
        rewards = [np.array([float(r)], dtype=np.float32) for r in rate]
        dones = [False] * self.agent_num
        infos = []
        for i in range(self.agent_num):
            iw = max(float(inter_w[i]), 1e-12)
            inter_dbm = 10.0 * math.log10(iw / 0.001)
            infos.append({'inst_rate': float(rate[i]), 'interference_dbm': inter_dbm})
        return (obs, rewards, dones, infos)

    def _reset_or_build_scenario(self):
        (ChannelConfig, generate_channel, _, _) = _maybe_import_channel_api()
        if generate_channel:
            scn = generate_channel(ChannelConfig(M=self.agent_num, N=self.ap_num, map_size=self.map_size, ap_min_distance=self.ap_min_distance, backhaul_cap_range=self.backhaul_cap_range, seed=int(self._rng.integers(0, 2 ** 31 - 1))))
            (self.users_xy, self.aps_xy) = (scn['users'], scn['aps'])
            (self.backhaul_caps, self.mu) = (scn['caps'], scn['mu'])
            self.distance = self._pairwise_dist(self.users_xy, self.aps_xy)
            return
        self.users_xy = self._rng.random((self.agent_num, 2)) * self.map_size
        self.aps_xy = self._rng.random((self.ap_num, 2)) * self.map_size
        self.backhaul_caps = self._rng.uniform(self.DEFAULT_BACKHAUL_RANGE[0], self.DEFAULT_BACKHAUL_RANGE[1], size=self.ap_num)
        self.distance = self._pairwise_dist(self.users_xy, self.aps_xy)
        self.mu = self._make_pathloss_with_shadow(self.distance, self._rng)

    def _pairwise_dist(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        return np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(axis=2))

    def _make_pathloss_with_shadow(self, dist: np.ndarray, rng) -> np.ndarray:
        (M, N) = dist.shape
        a = rng.random((1, N))
        b = rng.random((1, M))
        delta = rng.random()
        theta = np.zeros((M, N))
        for i in range(M):
            for j in range(N):
                theta[i, j] = math.sqrt(delta) * a[0, j] + math.sqrt(1 - delta) * b[0, i]
        f = 1.9 * 10 ** 9
        h_ap = 15
        h_u = 1.65
        L = 46.3 + 33.9 * np.log10(f) - 13.82 * np.log10(h_ap) - (1.11 * np.log10(f) - 0.7) * h_u + 1.56 * np.log10(f) - 0.8
        (d0, d1) = (10.0, 50.0)
        mu_db = np.zeros_like(dist)
        for i in range(M):
            for j in range(N):
                d = dist[i, j]
                if d > d1:
                    mu_db[i, j] = -L - 35 * np.log10(d / 1000.0) + 0.5 * theta[i, j]
                elif d0 <= d <= d1:
                    mu_db[i, j] = -L - 10 * np.log10(d1 ** 1.5 * d ** 2 / 1000.0 ** 2)
                else:
                    mu_db[i, j] = -L - 10 * np.log10(d1 ** 1.5 * d0 ** 2 / 1000.0 ** 2)
        return 10 ** (mu_db / 10.0)

    def _observe_initial(self) -> List[np.ndarray]:
        obs = []
        zeros_caps = np.zeros(self.cluster_max, dtype=np.float32)
        for _ in range(self.agent_num):
            obs.append(np.concatenate([[0.0], zeros_caps]).astype(np.float32))
        return obs

    def _observe_from_state(self, sinr: np.ndarray, ap_load: np.ndarray) -> List[np.ndarray]:
        obs = []
        for i in range(self.agent_num):
            row = self.mu[i, :]
            top = row.argsort()[-self.cluster_max:][::-1]
            caps_vec = ap_load[top] if ap_load is not None else np.zeros(self.cluster_max, dtype=float)
            o = np.concatenate([[float(sinr[i])], caps_vec.astype(np.float32)], axis=0).astype(np.float32)
            obs.append(o)
        return obs

    def _decode_actions(self, actions: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        M = self.agent_num
        mode_idx = np.zeros(M, dtype=np.int32)
        cluster_k = np.zeros(M, dtype=np.int32)
        size_options = np.array([3, 4, 5], dtype=np.int32)
        for i in range(M):
            a = np.asarray(actions[i]).reshape(-1)
            if a.size >= 6:
                (m_oh, s_oh) = (a[:3], a[3:6])
                mode_idx[i] = int(np.argmax(m_oh))
                s_idx = int(np.argmax(s_oh))
                allowed = size_options[size_options <= self.cluster_max]
                if allowed.size == 0:
                    allowed = np.array([min(3, self.cluster_max)], dtype=np.int32)
                s_idx = min(s_idx, allowed.size - 1)
                cluster_k[i] = allowed[s_idx]
            else:
                mode_idx[i] = 2
                cluster_k[i] = 1
        return (mode_idx, cluster_k)

    def _compute_rate(self, mode_idx: np.ndarray, cluster_k: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        (M, N) = (self.agent_num, self.ap_num)
        mu = self.mu
        pf = self.power_w
        W = self.bandwidth_hz
        clusters = []
        for i in range(M):
            K = int(cluster_k[i])
            row = mu[i, :]
            top = row.argsort()[-K:][::-1]
            clusters.append(top)
        sinr_num = np.zeros(M, dtype=float)
        inter_w = np.zeros(M, dtype=float)
        for i in range(M):
            mode = int(mode_idx[i])
            K = int(cluster_k[i])
            C = set((int(x) for x in clusters[i]))
            if mode == 0:
                for j in range(N):
                    p = pf / K if j in C else pf
                    g = self.gains['jt'] if j in C else self.gains['base']
                    contrib = p * g * mu[i, j]
                    if j in C:
                        sinr_num[i] += contrib
                    else:
                        inter_w[i] += contrib
            elif mode == 1:
                main_j = int(clusters[i][0])
                for j in range(N):
                    if j == main_j:
                        sinr_num[i] += pf * self.gains['cb_main'] * mu[i, j]
                    else:
                        inter_w[i] += pf * self.gains['cb_others'] * mu[i, j]
            else:
                main_j = int(clusters[i][0])
                for j in range(N):
                    if j == main_j:
                        sinr_num[i] += pf * self.gains['noncomp'] * mu[i, j]
                    else:
                        inter_w[i] += pf * self.gains['base'] * mu[i, j]
            inter_w[i] = max(inter_w[i], 1e-12)
        sinr = sinr_num / inter_w
        rate = W * np.log2(1.0 + sinr)
        served = [[] for _ in range(N)]
        for i in range(M):
            if mode_idx[i] == 0:
                for j in clusters[i]:
                    served[int(j)].append(i)
            else:
                served[int(clusters[i][0])].append(i)
        ap_load = np.zeros(N, dtype=float)
        for j in range(N):
            if not served[j]:
                continue
            load = float(rate[served[j]].sum())
            ap_load[j] = load
            cap_j = float(self.backhaul_caps[j])
            if load != 0 and load * self.omega > cap_j:
                scale = cap_j / (load * self.omega)
                for i in served[j]:
                    rate[i] *= scale
                ap_load[j] = cap_j
        self._last_inst_rate = rate.copy()
        self._last_interf_w = inter_w.copy()
        return (sinr, rate, inter_w, ap_load)
