from __future__ import annotations
import math
from itertools import combinations
from typing import Dict, List, Optional, Sequence, Tuple
import numpy as np
from gym import spaces
from envs.env_core import EnvCore

def encode_iicms_action(env: 'IICMSMobileEnvCore', mode: int, cluster_size: int, subset_index: int=0) -> np.ndarray:
    target = (int(mode), 1 if int(mode) == 2 else int(cluster_size))
    matches = [index for (index, action) in enumerate(env.joint_actions) if action == target]
    if not matches:
        raise ValueError(f'Invalid joint action: {target}')
    if not 0 <= int(subset_index) < len(matches):
        raise ValueError(f'Invalid subset_index {subset_index} for {target}; available={len(matches)}')
    action_index = matches[int(subset_index)]
    action = np.zeros(len(env.joint_actions), dtype=np.float32)
    action[action_index] = 1.0
    return action

class IICMSMobileEnvCore(EnvCore):
    MODES = {'JT': 0, 'CB': 1, 'NC': 2}

    def __init__(self, mobility: str='rwp', ue_speed: float=10.0, slot_dt: float=0.5, pathloss_model: str='legacy', csi_aging: bool=False, carrier_ghz: float=1.9, csi_block_slots: int=200, csi_pilot_slots: int=10, csi_slot_ms: float=1.0, size_options: Sequence[int]=(2, 3, 4, 5), reward_mode: str='mean_se', objective_horizon: int=100, reward_phi1: float=0.5, reward_phi2: float=1.0, rate_req_lo_mbps: float=40.0, rate_req_hi_mbps: float=40.0, min_ue_se_bps_hz: float=0.0, qos_penalty_weight: float=20.0, backhaul_jt_model: str='bottleneck', fronthaul_queue: int=0, fronthaul_buffer_s: float=1.0, ap_min_distance: float=50.0, reset_ues: bool=True, cluster_action_mode: str='topk', cluster_policy: str='dynamic', obs_include_xy: bool=False, obs_frames: int=1, obs_neighbor_ues: int=0, obs_ap_competition: int=0):
        super().__init__()
        self.mobility = str(mobility)
        self.ue_speed = float(ue_speed)
        self.slot_dt = float(slot_dt)
        self.pathloss_model = str(pathloss_model)
        self.csi_aging = bool(csi_aging)
        self.carrier_ghz = float(carrier_ghz)
        self.csi_block_slots = max(1, int(csi_block_slots))
        self.csi_pilot_slots = max(0, min(int(csi_pilot_slots), self.csi_block_slots - 1))
        self.csi_slot_ms = float(csi_slot_ms)
        self.size_options = self._validate_size_options(size_options)
        self.reward_mode = str(reward_mode)
        self.objective_horizon = max(1, int(objective_horizon))
        self.reward_phi1 = float(reward_phi1)
        self.reward_phi2 = float(reward_phi2)
        self.rate_req_lo_mbps = float(rate_req_lo_mbps)
        self.rate_req_hi_mbps = float(rate_req_hi_mbps)
        self.min_ue_se_bps_hz = max(float(min_ue_se_bps_hz), 0.0)
        self.qos_penalty_weight = float(qos_penalty_weight)
        self.backhaul_jt_model = str(backhaul_jt_model)
        self.fronthaul_queue = bool(fronthaul_queue)
        self.fronthaul_buffer_s = max(float(fronthaul_buffer_s), float(self.slot_dt))
        self.ap_min_distance = float(ap_min_distance)
        self.reset_ues = bool(reset_ues)
        self.obs_include_xy = bool(obs_include_xy)
        self.obs_frames = max(1, int(obs_frames))
        self.obs_neighbor_ues = max(0, int(obs_neighbor_ues))
        self.obs_ap_competition = bool(obs_ap_competition)
        cluster_action_mode = str(cluster_action_mode).lower()
        if cluster_action_mode not in {'topk', 'subset'}:
            raise ValueError("cluster_action_mode must be 'topk' or 'subset'")
        self.cluster_action_mode = cluster_action_mode
        self.cluster_policy = str(cluster_policy).lower()
        if self.cluster_policy not in {'static', 'dynamic'}:
            raise ValueError("cluster_policy must be 'static' or 'dynamic'")
        self._static_candidate_order: Optional[np.ndarray] = None
        self._shadow_gain = None
        self._waypoints = np.zeros((self.agent_num, 2), dtype=float)
        self._velocity = np.zeros((self.agent_num, 2), dtype=float)
        self._speed = np.zeros(self.agent_num, dtype=float)
        self._rate_req_mbps = np.zeros(self.agent_num, dtype=float)
        self._previous_clusters: List[Optional[np.ndarray]] = [None] * self.agent_num
        self._previous_primary = np.full(self.agent_num, -1, dtype=np.int32)
        self._previous_mode = np.full(self.agent_num, -1, dtype=np.int32)
        self._t = 0
        self._rho2 = np.ones(self.agent_num, dtype=float)
        self._pilot_data_fraction = 1.0
        self._last_step_summary: Dict[str, float] = {}
        self._fronthaul_queue_bits = np.zeros(self.ap_num, dtype=float)
        self._obs_history: Optional[np.ndarray] = None
        self.cluster_changes = np.zeros(self.agent_num, dtype=np.int64)
        self.overload_events = np.zeros(self.agent_num, dtype=np.int64)
        self._rebuild_action_space()
        super().configure(ap_min_distance=self.ap_min_distance)
        self._capture_scenario()
        self._validate_ap_spacing()
        self._prepare_episode()

    @staticmethod
    def _validate_size_options(sizes: Sequence[int]) -> Tuple[int, ...]:
        values = tuple(sorted({int(value) for value in sizes}))
        if not values or values[0] < 1:
            raise ValueError('size_options must contain positive integers')
        return values

    def _rebuild_action_space(self) -> None:
        self.cluster_max = int(max(self.size_options))
        cluster_choices = []
        for size in self.size_options:
            if self.cluster_action_mode == 'topk':
                cluster_choices.append((int(size), tuple(range(int(size))), f'L{int(size)}'))
                continue
            for (subset_index, subset) in enumerate(combinations(range(self.cluster_max), int(size))):
                cluster_choices.append((int(size), tuple((int(rank) for rank in subset)), f'L{int(size)}_S{subset_index:02d}'))
        if not cluster_choices:
            raise ValueError('No cooperative-cluster actions were generated')
        self.cluster_choice_labels = [label for (_size, _ranks, label) in cluster_choices]
        self.cluster_choice_ranks = [ranks for (_size, ranks, _label) in cluster_choices]
        self.cluster_choice_sizes = [size for (size, _ranks, _label) in cluster_choices]
        self.joint_num_cluster_choices = len(cluster_choices)
        self.joint_actions = []
        self.action_names = []
        self.joint_action_cluster_choice = []
        self.joint_action_cluster_ranks = []
        for mode in (0, 1):
            mode_name = ('JT', 'CB')[mode]
            for (choice_index, (size, ranks, label)) in enumerate(cluster_choices):
                self.joint_actions.append((mode, size))
                self.action_names.append(f'{mode_name}_{label}')
                self.joint_action_cluster_choice.append(choice_index)
                self.joint_action_cluster_ranks.append(ranks)
        self.joint_actions.append((2, 1))
        self.action_names.append('NC')
        self.joint_action_cluster_choice.append(0)
        self.joint_action_cluster_ranks.append((0,))
        self.joint_valid_flat_indices = [int(mode) * self.joint_num_cluster_choices + choice for ((mode, _size), choice) in zip(self.joint_actions, self.joint_action_cluster_choice)]
        if self.obs_neighbor_ues >= self.agent_num:
            raise ValueError('obs_neighbor_ues must be smaller than num_agents')
        if self.obs_neighbor_ues and not (self.obs_include_xy and self.obs_ap_competition):
            raise ValueError('obs_neighbor_ues requires obs_include_xy=1 and obs_ap_competition=1')
        self.obs_token_dim = (
            (6 if self.obs_include_xy else 4)
            + (1 if self.fronthaul_queue else 0)
            + (2 if self.obs_ap_competition else 0)
        )
        self.obs_tokens_per_frame = 1 + self.obs_neighbor_ues + self.cluster_max
        self.obs_frame_dim = self.obs_token_dim * self.obs_tokens_per_frame
        self.obs_dim = self.obs_frame_dim * self.obs_frames
        self.action_dim = [len(self.joint_actions)]

    def set_size_options(self, sizes: Sequence[int]) -> None:
        self.size_options = self._validate_size_options(sizes)
        self._rebuild_action_space()

    def configure(self, num_aps: Optional[int]=None, num_agents: Optional[int]=None, map_size: Optional[float]=None, ap_min_distance: Optional[float]=None, antennas_per_ap: Optional[int]=None, channel_file: Optional[str]=None, seed: Optional[int]=None, backhaul_cap_range: Optional[Tuple[float, float]]=None, size_options: Optional[Sequence[int]]=None, mobility: Optional[str]=None, ue_speed: Optional[float]=None, slot_dt: Optional[float]=None, pathloss_model: Optional[str]=None, csi_aging: Optional[int]=None, carrier_ghz: Optional[float]=None, csi_block_slots: Optional[int]=None, csi_pilot_slots: Optional[int]=None, csi_slot_ms: Optional[float]=None, reward_mode: Optional[str]=None, objective_horizon: Optional[int]=None, reward_phi1: Optional[float]=None, reward_phi2: Optional[float]=None, rate_req_lo_mbps: Optional[float]=None, rate_req_hi_mbps: Optional[float]=None, min_ue_se_bps_hz: Optional[float]=None, qos_penalty_weight: Optional[float]=None, backhaul_jt_model: Optional[str]=None, fronthaul_queue: Optional[int]=None, fronthaul_buffer_s: Optional[float]=None, reset_ues: Optional[int]=None, cluster_action_mode: Optional[str]=None, cluster_policy: Optional[str]=None, obs_include_xy: Optional[int]=None, obs_frames: Optional[int]=None, obs_neighbor_ues: Optional[int]=None, obs_ap_competition: Optional[int]=None) -> None:
        if mobility is not None:
            self.mobility = str(mobility)
        if ue_speed is not None:
            self.ue_speed = float(ue_speed)
        if slot_dt is not None:
            self.slot_dt = float(slot_dt)
        if pathloss_model is not None:
            self.pathloss_model = str(pathloss_model)
        if csi_aging is not None:
            self.csi_aging = bool(csi_aging)
        if carrier_ghz is not None:
            self.carrier_ghz = float(carrier_ghz)
        if csi_block_slots is not None:
            self.csi_block_slots = max(1, int(csi_block_slots))
        if csi_pilot_slots is not None:
            self.csi_pilot_slots = max(0, min(int(csi_pilot_slots), self.csi_block_slots - 1))
        if csi_slot_ms is not None:
            self.csi_slot_ms = float(csi_slot_ms)
        if size_options is not None:
            self.size_options = self._validate_size_options(size_options)
        if cluster_action_mode is not None:
            value = str(cluster_action_mode).lower()
            if value not in {'topk', 'subset'}:
                raise ValueError("cluster_action_mode must be 'topk' or 'subset'")
            self.cluster_action_mode = value
        if cluster_policy is not None:
            value = str(cluster_policy).lower()
            if value not in {'static', 'dynamic'}:
                raise ValueError("cluster_policy must be 'static' or 'dynamic'")
            self.cluster_policy = value
        if reward_mode is not None:
            self.reward_mode = str(reward_mode)
        if objective_horizon is not None:
            self.objective_horizon = max(1, int(objective_horizon))
        if reward_phi1 is not None:
            self.reward_phi1 = float(reward_phi1)
        if reward_phi2 is not None:
            self.reward_phi2 = float(reward_phi2)
        if rate_req_lo_mbps is not None:
            self.rate_req_lo_mbps = float(rate_req_lo_mbps)
        if rate_req_hi_mbps is not None:
            self.rate_req_hi_mbps = float(rate_req_hi_mbps)
        if min_ue_se_bps_hz is not None:
            self.min_ue_se_bps_hz = max(float(min_ue_se_bps_hz), 0.0)
        if qos_penalty_weight is not None:
            self.qos_penalty_weight = float(qos_penalty_weight)
        if backhaul_jt_model is not None:
            self.backhaul_jt_model = str(backhaul_jt_model)
        if fronthaul_queue is not None:
            self.fronthaul_queue = bool(fronthaul_queue)
        if fronthaul_buffer_s is not None:
            self.fronthaul_buffer_s = max(float(fronthaul_buffer_s), float(self.slot_dt))
        if reset_ues is not None:
            self.reset_ues = bool(reset_ues)
        if obs_include_xy is not None:
            self.obs_include_xy = bool(obs_include_xy)
        if obs_frames is not None:
            self.obs_frames = max(1, int(obs_frames))
        if obs_neighbor_ues is not None:
            self.obs_neighbor_ues = max(0, int(obs_neighbor_ues))
        if obs_ap_competition is not None:
            self.obs_ap_competition = bool(obs_ap_competition)
        if self.mobility not in {'static', 'rwp', 'random_walk'}:
            raise ValueError(f'Unknown mobility model: {self.mobility}')
        if self.pathloss_model not in {'legacy', 'classic'}:
            raise ValueError(f'Unknown pathloss model: {self.pathloss_model}')
        if self.backhaul_jt_model not in {'product', 'bottleneck'}:
            raise ValueError(f'Unknown backhaul_jt_model: {self.backhaul_jt_model}')
        super().configure(num_aps=num_aps, num_agents=num_agents, map_size=map_size, ap_min_distance=ap_min_distance, antennas_per_ap=antennas_per_ap, channel_file=channel_file, seed=seed, backhaul_cap_range=backhaul_cap_range)
        self._rebuild_action_space()
        self._capture_scenario()
        self._validate_ap_spacing()
        self._prepare_episode()

    def load_channel(self, path_or_dict) -> bool:
        ok = super().load_channel(path_or_dict)
        if ok:
            self._rebuild_action_space()
            self._capture_scenario()
            self._validate_ap_spacing()
            self._prepare_episode()
        return ok

    @staticmethod
    def _pathloss_constants() -> float:
        f_mhz = 1900.0
        (h_ap, h_ue) = (15.0, 1.65)
        return 46.3 + 33.9 * np.log10(f_mhz) - 13.82 * np.log10(h_ap) - (1.11 * np.log10(f_mhz) - 0.7) * h_ue + 1.56 * np.log10(f_mhz) - 0.8

    def _pathloss_db_legacy(self, distance: np.ndarray) -> np.ndarray:
        dist = np.maximum(np.asarray(distance, dtype=float), 0.001)
        loss_const = self._pathloss_constants()
        (d0, d1) = (10.0, 50.0)
        out = np.empty_like(dist)
        far = dist > d1
        middle = (dist >= d0) & ~far
        near = ~(far | middle)
        out[far] = -loss_const - 35.0 * np.log10(dist[far] / 1000.0)
        out[middle] = -loss_const - 10.0 * np.log10(d1 ** 1.5 * dist[middle] / 1000.0 ** 2)
        out[near] = -loss_const - 10.0 * np.log10(d1 ** 1.5 * d0 ** 2 / 1000.0 ** 2)
        return out

    def _pathloss_db_classic(self, distance: np.ndarray) -> np.ndarray:
        dist_km = np.maximum(np.asarray(distance, dtype=float), 0.001) / 1000.0
        loss_const = self._pathloss_constants()
        (d0_km, d1_km) = (0.01, 0.05)
        out = np.empty_like(dist_km)
        far = dist_km > d1_km
        middle = (dist_km >= d0_km) & ~far
        near = ~(far | middle)
        out[far] = -loss_const - 35.0 * np.log10(dist_km[far])
        out[middle] = -loss_const - 15.0 * np.log10(d1_km) - 20.0 * np.log10(dist_km[middle])
        out[near] = -loss_const - 15.0 * np.log10(d1_km) - 20.0 * np.log10(d0_km)
        return out

    def _pathloss_gain(self, distance: np.ndarray) -> np.ndarray:
        if self.pathloss_model == 'legacy':
            loss_db = self._pathloss_db_legacy(distance)
        else:
            loss_db = self._pathloss_db_classic(distance)
        return 10.0 ** (loss_db / 10.0)

    def _capture_scenario(self) -> None:
        self.distance = self._pairwise_dist(self.users_xy, self.aps_xy)
        base_gain = self._pathloss_gain(self.distance)
        self._shadow_gain = np.asarray(self.mu / np.maximum(base_gain, 1e-30), dtype=float)

    def _validate_ap_spacing(self) -> None:
        if self.ap_num < 2 or self.ap_min_distance <= 0.0:
            return
        pair_distance = np.linalg.norm(self.aps_xy[:, None, :] - self.aps_xy[None, :, :], axis=2)
        pair_distance[np.diag_indices(self.ap_num)] = np.inf
        minimum = float(np.min(pair_distance))
        if minimum + 1e-09 < self.ap_min_distance:
            raise ValueError(f'AP spacing constraint violated: minimum={minimum:.3f} m, required={self.ap_min_distance:.3f} m')

    def _sample_episode_positions(self) -> None:
        self.users_xy = self._rng.random((self.agent_num, 2)) * self.map_size
        self.distance = self._pairwise_dist(self.users_xy, self.aps_xy)
        self.mu = self._pathloss_gain(self.distance) * self._shadow_gain

    @staticmethod
    def _bessel_j0(x: np.ndarray) -> np.ndarray:
        x_abs = np.abs(np.asarray(x, dtype=float))
        out = np.empty_like(x_abs)
        small = x_abs < 8.0
        y = x_abs[small] ** 2
        ans1 = 57568490574.0 + y * (-13362590354.0 + y * (651619640.7 + y * (-11214424.18 + y * (77392.33017 + y * -184.9052456))))
        ans2 = 57568490411.0 + y * (1029532985.0 + y * (9494680.718 + y * (59272.64853 + y * (267.8532712 + y))))
        out[small] = ans1 / ans2
        out[x_abs == 0.0] = 1.0
        large = ~small
        z = 8.0 / x_abs[large]
        y = z ** 2
        xx = x_abs[large] - 0.785398164
        ans1 = 1.0 + y * (-0.001098628627 + y * (2.734510407e-05 + y * (-2.073370639e-06 + y * 2.093887211e-07)))
        ans2 = -0.01562499995 + y * (0.0001430488765 + y * (-6.911147651e-06 + y * (7.621095161e-07 - y * 9.34935152e-08)))
        out[large] = np.sqrt(0.636619772 / x_abs[large]) * (np.cos(xx) * ans1 - z * np.sin(xx) * ans2)
        return out

    def _refresh_aging(self) -> None:
        if not self.csi_aging:
            self._rho2 = np.ones(self.agent_num, dtype=float)
            self._pilot_data_fraction = 1.0
            return
        data_slots = self.csi_block_slots - self.csi_pilot_slots
        self._pilot_data_fraction = float(data_slots) / float(self.csi_block_slots)
        ages_s = np.arange(data_slots, dtype=float) * self.csi_slot_ms / 1000.0
        doppler_hz = self.carrier_ghz * 1000000000.0 * self._speed / 299792458.0
        argument = 2.0 * np.pi * doppler_hz[:, None] * ages_s[None, :]
        self._rho2 = np.mean(self._bessel_j0(argument) ** 2, axis=1)

    def _sample_ue_speeds(self) -> None:
        self._speed = np.full(self.agent_num, self.ue_speed if self.mobility != 'static' else 0.0, dtype=float)

    def _prepare_episode(self) -> None:
        self._sample_ue_speeds()
        self._rate_req_mbps = self._rng.uniform(self.rate_req_lo_mbps, self.rate_req_hi_mbps, size=self.agent_num)
        self._waypoints = self._rng.random((self.agent_num, 2)) * self.map_size
        delta = self._waypoints - self.users_xy
        norm = np.linalg.norm(delta, axis=1)
        self._velocity = np.zeros_like(delta)
        moving = norm > 1e-12
        self._velocity[moving] = delta[moving] / norm[moving, None] * self._speed[moving, None]
        angle = self._rng.uniform(0.0, 2.0 * np.pi, size=self.agent_num)
        self._velocity[~moving, 0] = np.cos(angle[~moving]) * self._speed[~moving]
        self._velocity[~moving, 1] = np.sin(angle[~moving]) * self._speed[~moving]
        self._refresh_aging()
        self._fronthaul_queue_bits = np.zeros(self.ap_num, dtype=float)
        self._previous_clusters = [None] * self.agent_num
        self._previous_primary = np.full(self.agent_num, -1, dtype=np.int32)
        self._previous_mode = np.full(self.agent_num, -1, dtype=np.int32)
        if self.cluster_policy == 'static':
            self._static_candidate_order = self._current_candidate_order().copy()
        else:
            self._static_candidate_order = None
        self._obs_history = np.zeros((self.agent_num, self.obs_frames, self.obs_frame_dim), dtype=np.float32)
        self.cluster_changes = np.zeros(self.agent_num, dtype=np.int64)
        self.overload_events = np.zeros(self.agent_num, dtype=np.int64)
        self._t = 0

    def _advance_positions(self) -> None:
        if self.mobility == 'static' or np.max(self._speed) <= 0.0:
            return
        if self.mobility == 'random_walk':
            angle = np.arctan2(self._velocity[:, 1], self._velocity[:, 0]) + self._rng.normal(0.0, 0.25, size=self.agent_num)
            self._velocity = np.column_stack((np.cos(angle), np.sin(angle))) * self._speed[:, None]
            self.users_xy = self.users_xy + self._velocity * self.slot_dt
            low = self.users_xy < 0.0
            high = self.users_xy > self.map_size
            self.users_xy[low] *= -1.0
            self.users_xy[high] = 2.0 * self.map_size - self.users_xy[high]
            self._velocity[low | high] *= -1.0
            return
        delta = self._waypoints - self.users_xy
        norm = np.linalg.norm(delta, axis=1)
        step = self._speed * self.slot_dt
        reached = (norm <= step) | (norm < 1e-12)
        moving = ~reached
        if np.any(moving):
            direction = delta[moving] / norm[moving, None]
            self.users_xy[moving] += direction * step[moving, None]
        if np.any(reached):
            rows = np.flatnonzero(reached)
            self._waypoints[rows] = self._rng.random((rows.size, 2)) * self.map_size
            delta[rows] = self._waypoints[rows] - self.users_xy[rows]
            norm[rows] = np.linalg.norm(delta[rows], axis=1)
            nonzero = rows[norm[rows] > 1e-12]
            self._velocity[nonzero] = delta[nonzero] / norm[nonzero, None] * self._speed[nonzero, None]

    def _update_channels(self) -> None:
        self.distance = self._pairwise_dist(self.users_xy, self.aps_xy)
        self.mu = self._pathloss_gain(self.distance) * self._shadow_gain

    def reset(self) -> List[np.ndarray]:
        if self.reset_ues:
            self._sample_episode_positions()
        self._prepare_episode()
        return self._observe_initial()

    def _decode_iicms_actions(self, actions: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        mode_idx = np.zeros(self.agent_num, dtype=np.int32)
        cluster_size = np.zeros(self.agent_num, dtype=np.int32)
        action_idx = np.zeros(self.agent_num, dtype=np.int64)
        for (index, action) in enumerate(actions):
            flat = np.asarray(action).reshape(-1)
            if flat.size != len(self.joint_actions):
                raise ValueError(f'Each action must be one joint-action one-hot vector ({len(self.joint_actions)} values), got {flat.size}')
            action_index = int(np.argmax(flat))
            (mode_idx[index], cluster_size[index]) = self.joint_actions[action_index]
            action_idx[index] = action_index
        return (mode_idx, cluster_size, action_idx)

    def _clusters_for_size(self, cluster_size: np.ndarray) -> List[np.ndarray]:
        strongest = self._candidate_ap_order()
        clusters = []
        for index in range(self.agent_num):
            size = int(np.clip(cluster_size[index], 1, self.cluster_max))
            clusters.append(strongest[index, :size].copy())
        return clusters

    def _current_candidate_order(self) -> np.ndarray:
        order = np.argsort(self.mu, axis=1)
        return order[:, -self.cluster_max:][:, ::-1].copy()

    def _candidate_ap_order(self) -> np.ndarray:
        if self.cluster_policy == 'static' and self._static_candidate_order is not None:
            return self._static_candidate_order
        return self._current_candidate_order()

    def _clusters_for_actions(self, action_idx: np.ndarray) -> List[np.ndarray]:
        strongest = self._candidate_ap_order()
        clusters = []
        for (ue, action_index) in enumerate(action_idx):
            ranks = self.joint_action_cluster_ranks[int(action_index)]
            clusters.append(strongest[ue, np.asarray(ranks, dtype=int)].copy())
        return clusters

    def _compute_metrics(self, mode_idx: np.ndarray, cluster_size: np.ndarray, clusters: List[np.ndarray], record_overload: bool=True, advance_fronthaul_queue: bool=False) -> Dict[str, np.ndarray]:
        signal = np.zeros(self.agent_num, dtype=float)
        interference = np.zeros(self.agent_num, dtype=float)
        power = float(self.power_w)
        for ue in range(self.agent_num):
            selected = set((int(ap) for ap in clusters[ue]))
            primary = int(clusters[ue][0])
            mode = int(mode_idx[ue])
            size = len(clusters[ue])
            if mode == 0:
                for ap in range(self.ap_num):
                    in_cluster = ap in selected
                    contribution = power / size * (self.gains['jt'] if in_cluster else self.gains['base']) * self.mu[ue, ap]
                    if in_cluster:
                        signal[ue] += contribution
                    else:
                        interference[ue] += contribution
            elif mode == 1:
                for ap in range(self.ap_num):
                    if ap == primary:
                        signal[ue] += power * self.gains['cb_main'] * self.mu[ue, ap]
                    elif ap in selected:
                        interference[ue] += power * self.gains['cb_others'] * self.mu[ue, ap]
                    else:
                        interference[ue] += power * self.gains['base'] * self.mu[ue, ap]
            else:
                for ap in range(self.ap_num):
                    if ap == primary:
                        signal[ue] += power * self.gains['noncomp'] * self.mu[ue, ap]
                    else:
                        interference[ue] += power * self.gains['base'] * self.mu[ue, ap]
        interference = np.maximum(interference, 1e-12)
        sinr_clean = signal / interference
        sinr = self._rho2 * sinr_clean
        rate_pre = self.bandwidth_hz * np.log2(1.0 + sinr) * self._pilot_data_fraction
        offered_load = np.zeros(self.ap_num, dtype=float)
        served_by_ap: List[List[int]] = [[] for _ in range(self.ap_num)]
        for ue in range(self.agent_num):
            if mode_idx[ue] == 0:
                for ap in clusters[ue]:
                    served_by_ap[int(ap)].append(ue)
            else:
                served_by_ap[int(clusters[ue][0])].append(ue)
        served_by_ap_matrix = np.zeros((self.agent_num, self.ap_num), dtype=bool)
        for (ap, users) in enumerate(served_by_ap):
            if not users:
                continue
            offered_load[ap] = float(rate_pre[users].sum())
            for ue in users:
                served_by_ap_matrix[ue, ap] = True
        fronthaul = self._apply_fronthaul_constraint(offered_load, advance_queue=advance_fronthaul_queue)
        ap_scale = fronthaul['ap_scale']
        if record_overload:
            overloaded_aps = fronthaul['overloaded']
            for ue in range(self.agent_num):
                if np.any(served_by_ap_matrix[ue] & overloaded_aps):
                    self.overload_events[ue] += 1
        backhaul_scale = np.ones(self.agent_num, dtype=float)
        for ue in range(self.agent_num):
            serving_scales = ap_scale[served_by_ap_matrix[ue]]
            if serving_scales.size == 0:
                continue
            if self.backhaul_jt_model == 'product':
                backhaul_scale[ue] = float(np.prod(serving_scales))
            else:
                backhaul_scale[ue] = float(np.min(serving_scales))
        rate_after_backhaul = rate_pre * backhaul_scale
        return {'signal': signal, 'interference': interference, 'sinr_clean': sinr_clean, 'sinr': sinr, 'rate_pre': rate_pre, 'rate_bh': rate_after_backhaul, 'rate': rate_after_backhaul, 'offered_load': offered_load, 'effective_load': fronthaul['effective_load'], 'backhaul_scale': backhaul_scale, 'ap_scale': ap_scale, 'backhaul_jt_model': self.backhaul_jt_model, 'served_by_ap': served_by_ap_matrix, **{key: value for (key, value) in fronthaul.items() if key not in {'ap_scale', 'effective_load'}}}

    def _apply_fronthaul_constraint(self, offered_load: np.ndarray, advance_queue: bool=False) -> Dict[str, np.ndarray]:
        offered = np.maximum(np.asarray(offered_load, dtype=float), 0.0)
        capacity = np.maximum(np.asarray(self.backhaul_caps, dtype=float), 0.0)
        service_bits = capacity * self.slot_dt
        arrival_bits = offered * self.omega * self.slot_dt
        queue_capacity_bits = capacity * self.fronthaul_buffer_s
        queue_before = np.asarray(self._fronthaul_queue_bits, dtype=float).copy()
        safe_arrival = np.maximum(arrival_bits, 1e-30)
        if not self.fronthaul_queue:
            served_bits = np.minimum(arrival_bits, service_bits)
            queue_after = np.zeros(self.ap_num, dtype=float)
            overflow_bits = np.zeros(self.ap_num, dtype=float)
            current_overflow_bits = np.zeros(self.ap_num, dtype=float)
            old_service_bits = np.zeros(self.ap_num, dtype=float)
        else:
            queue_before = np.minimum(queue_before, queue_capacity_bits)
            total_queued = queue_before + arrival_bits
            service_available = np.minimum(total_queued, service_bits)
            fresh_share = np.divide(arrival_bits, np.maximum(total_queued, 1e-30), out=np.zeros(self.ap_num, dtype=float), where=total_queued > 0.0)
            served_bits = service_available * fresh_share
            old_service_bits = service_available - served_bits
            remaining_after_service = total_queued - service_available
            overflow_bits = np.maximum(remaining_after_service - queue_capacity_bits, 0.0)
            current_overflow_bits = overflow_bits * fresh_share
            queue_after = np.minimum(remaining_after_service, queue_capacity_bits)
        ap_scale = np.divide(served_bits, safe_arrival, out=np.ones(self.ap_num, dtype=float), where=arrival_bits > 0.0)
        ap_scale = np.clip(ap_scale, 0.0, 1.0)
        effective_load = np.divide(served_bits, max(self.omega, 1e-30) * self.slot_dt, out=np.zeros(self.ap_num, dtype=float))
        overloaded = arrival_bits > service_bits + 1e-12
        service_used = old_service_bits + served_bits
        utilization = np.divide(service_used, service_bits, out=np.zeros(self.ap_num, dtype=float), where=service_bits > 0.0)
        drop_ratio = np.divide(current_overflow_bits, safe_arrival, out=np.zeros(self.ap_num, dtype=float), where=arrival_bits > 0.0)
        queue_ratio = np.divide(queue_after, queue_capacity_bits, out=np.zeros(self.ap_num, dtype=float), where=queue_capacity_bits > 0.0)
        queue_delay_ms = np.divide(queue_after, capacity, out=np.zeros(self.ap_num, dtype=float), where=capacity > 0.0) * 1000.0
        if self.fronthaul_queue and advance_queue:
            self._fronthaul_queue_bits = queue_after.copy()
        return {'ap_scale': ap_scale, 'effective_load': effective_load, 'fronthaul_queue_before': queue_before, 'fronthaul_queue_bits': queue_after, 'fronthaul_queue_capacity_bits': queue_capacity_bits, 'fronthaul_queue_ratio': np.clip(queue_ratio, 0.0, 1.0), 'fronthaul_arrival_bits': arrival_bits, 'fronthaul_served_bits': served_bits, 'fronthaul_overflow_bits': overflow_bits, 'fronthaul_current_drop_bits': current_overflow_bits, 'fronthaul_drop_ratio': np.clip(drop_ratio, 0.0, 1.0), 'fronthaul_service_utilization': np.clip(utilization, 0.0, 1.0), 'fronthaul_queue_delay_ms': np.maximum(queue_delay_ms, 0.0), 'overloaded': overloaded}

    def _update_cluster_changes(self, clusters: List[np.ndarray]) -> np.ndarray:
        changed = np.zeros(self.agent_num, dtype=bool)
        for (ue, cluster) in enumerate(clusters):
            previous = self._previous_clusters[ue]
            if previous is not None:
                changed[ue] = set(cluster.tolist()) != set(previous.tolist())
            self._previous_clusters[ue] = cluster.copy()
        self.cluster_changes += changed.astype(np.int64)
        return changed

    def _size_from_value(self, cluster_size: int) -> int:
        values = np.asarray(self.size_options, dtype=int)
        return int(values[np.argmin(np.abs(values - int(cluster_size)))])

    def evaluate_current(self, mode: int, cluster_size: int) -> Dict[str, object]:
        mode_idx = np.full(self.agent_num, int(mode), dtype=np.int32)
        size = self._size_from_value(cluster_size)
        cluster_k = np.full(self.agent_num, size, dtype=np.int32)
        clusters = self._clusters_for_size(cluster_k)
        metrics = self._compute_metrics(mode_idx, cluster_k, clusters, record_overload=False)
        rates = metrics['rate']
        overload = (metrics['offered_load'] > 0.0) & (metrics['offered_load'] * self.omega > self.backhaul_caps)
        return {'sum_rate_mbps': float(np.sum(rates)) / 1000000.0, 'per_ue_rate_mbps': rates / 1000000.0, 'overload_ratio': float(np.mean(overload)), 'backhaul_scale': float(np.mean(metrics['backhaul_scale'])), 'fronthaul_queue_ratio': float(np.mean(metrics['fronthaul_queue_ratio'])), 'fronthaul_drop_ratio': float(np.mean(metrics['fronthaul_drop_ratio'])), 'rho2': float(np.mean(self._rho2))}

    def _paper_reward(self, rates_bps: np.ndarray, mode_idx: np.ndarray, clusters: List[np.ndarray], offered_load: np.ndarray) -> np.ndarray:
        rate_mbps = rates_bps / 1000000.0
        satisfied = rate_mbps >= self._rate_req_mbps
        rewards = np.zeros(self.agent_num, dtype=float)
        for ue in range(self.agent_num):
            served = clusters[ue] if mode_idx[ue] == 0 else clusters[ue][:1]
            overloaded = 0.0
            for ap in served:
                overloaded += float(offered_load[int(ap)] * self.omega > self.backhaul_caps[int(ap)])
            network_rate_mbps = float(np.sum(rate_mbps))
            rewards[ue] = float(satisfied[ue]) * (network_rate_mbps - self.reward_phi1 * rate_mbps[ue]) - self.reward_phi2 * overloaded
        return rewards

    def _qos_terms(self, rates_bps: np.ndarray) -> Dict[str, np.ndarray]:
        rates_mbps = rates_bps / 1000000.0
        shortfall = np.maximum(self._rate_req_mbps - rates_mbps, 0.0)
        violation = shortfall > 0.0
        normalized_shortfall = shortfall / np.maximum(self._rate_req_mbps, 1e-12)
        penalty = np.mean(violation.astype(float) + normalized_shortfall)
        return {'rates_mbps': rates_mbps, 'shortfall_mbps': shortfall, 'violation': violation, 'normalized_shortfall': normalized_shortfall, 'penalty': np.asarray(penalty, dtype=float)}

    def _observe_initial(self) -> List[np.ndarray]:
        return [np.zeros(self.obs_dim, dtype=np.float32) for _ in range(self.agent_num)]

    def _observe_from_state(self, sinr: np.ndarray, ap_load: np.ndarray, effective_rate_bps: Optional[np.ndarray]=None, ap_scale: Optional[np.ndarray]=None, clusters: Optional[List[np.ndarray]]=None) -> List[np.ndarray]:
        n_aps = min(self.cluster_max, self.ap_num)
        order = self._candidate_ap_order()[:, :n_aps]
        candidate_mask = None
        candidate_count_token = None
        candidate_demand_token = None
        channel_correlation = None
        candidate_overlap = None
        neighbor_order = None
        if self.obs_ap_competition or self.obs_neighbor_ues:
            candidate_mask = np.zeros(
                (self.agent_num, self.ap_num), dtype=bool
            )
            candidate_mask[np.arange(self.agent_num)[:, None], order] = True
            candidate_counts_per_ap = candidate_mask.sum(axis=0)
            candidate_demand_per_ap = candidate_mask.T @ (
                self._rate_req_mbps * 1000000.0
            )
            candidate_count_token = np.clip(
                candidate_counts_per_ap[order]
                / float(max(1, self.agent_num)),
                0.0,
                1.0,
            )
            candidate_demand_token = np.clip(
                candidate_demand_per_ap[order]
                / np.maximum(self.backhaul_caps[order], 1.0),
                0.0,
                1.0,
            )
        if self.obs_neighbor_ues:
            channel_norm = np.linalg.norm(self.mu, axis=1, keepdims=True)
            channel_unit = self.mu / np.maximum(channel_norm, 1e-30)
            channel_correlation = np.clip(
                channel_unit @ channel_unit.T,
                0.0,
                1.0,
            )
            candidate_overlap = (
                candidate_mask.astype(float)
                @ candidate_mask.astype(float).T
                / float(max(1, n_aps))
            )
            conflict_score = (
                0.65 * candidate_overlap
                + 0.35 * channel_correlation
            )
            np.fill_diagonal(conflict_score, -np.inf)
            neighbor_order = np.argsort(
                conflict_score, axis=1
            )[:, -self.obs_neighbor_ues:][:, ::-1]
        top_strength = np.maximum(self.mu[np.arange(self.agent_num), order[:, 0]], 1e-30)
        top_mu = self.mu[np.arange(self.agent_num)[:, None], order]
        relative_gain = np.clip(10.0 * np.log10(np.maximum(top_mu, 1e-30) / top_strength[:, None]) / 30.0, -1.0, 0.0)
        safe_capacity = np.maximum(self.backhaul_caps[order], 1.0)
        load_ratio = np.clip(np.log2(1.0 + np.maximum(ap_load[order], 0.0) / safe_capacity) / 2.0, 0.0, 1.0)
        if ap_scale is None:
            ap_scale = np.ones(self.ap_num, dtype=float)
        scale_token = np.clip(ap_scale[order], 0.0, 1.0)
        served = np.zeros((self.agent_num, n_aps), dtype=float)
        if clusters is not None:
            for (ue, cluster) in enumerate(clusters):
                if cluster is None:
                    continue
                selected = set((int(ap) for ap in cluster))
                served[ue] = [float(int(ap) in selected) for ap in order[ue]]
        queue_capacity = np.maximum(np.asarray(self.backhaul_caps, dtype=float) * self.fronthaul_buffer_s, 1.0)
        queue_ratio_by_ap = np.clip(np.asarray(self._fronthaul_queue_bits, dtype=float) / queue_capacity, 0.0, 1.0)
        queue_ratio = queue_ratio_by_ap[order]
        if effective_rate_bps is None:
            effective_rate_mbps = np.zeros(self.agent_num, dtype=float)
        else:
            effective_rate_mbps = np.asarray(effective_rate_bps, dtype=float) / 1000000.0
        target_mbps = np.maximum(self._rate_req_mbps, 1.0)
        if self.obs_include_xy:
            ue_xy_token = np.clip(2.0 * self.users_xy[:self.agent_num] / self.map_size - 1.0, -1.0, 1.0)
            ap_xy_token = np.clip(2.0 * self.aps_xy[order] / self.map_size - 1.0, -1.0, 1.0)
        sinr_token = np.clip(np.log10(1.0 + np.maximum(sinr, 0.0)) / 3.0, 0.0, 1.0)
        rate_token = np.clip(effective_rate_mbps / (2.5 * target_mbps), 0.0, 2.0)
        qos_gap = np.clip((effective_rate_mbps - target_mbps) / target_mbps, -1.0, 3.0)
        target_token = np.clip(target_mbps / 100.0, 0.0, 1.0)
        previous_cluster_size = np.asarray([
            len(cluster) if cluster is not None else 0
            for cluster in self._previous_clusters
        ], dtype=float)
        previous_mode_token = np.clip(
            (self._previous_mode.astype(float) + 1.0) / 3.0,
            0.0,
            1.0,
        )
        previous_cluster_token = np.clip(
            previous_cluster_size / float(max(1, self.cluster_max)),
            0.0,
            1.0,
        )
        queue_pressure = np.zeros(self.agent_num, dtype=float)
        if self.fronthaul_queue and clusters is not None:
            for (ue, cluster) in enumerate(clusters):
                if cluster is None or len(cluster) == 0:
                    continue
                queue_pressure[ue] = float(np.max(queue_ratio_by_ap[np.asarray(cluster, dtype=int)]))
        observations = []
        for ue in range(self.agent_num):
            self_token = np.asarray([sinr_token[ue], rate_token[ue], qos_gap[ue], target_token[ue]])
            if self.obs_include_xy:
                self_token = np.concatenate((self_token, ue_xy_token[ue]))
            if self.fronthaul_queue:
                self_token = np.concatenate((self_token, [queue_pressure[ue]]))
            if self.obs_ap_competition:
                self_token = np.concatenate((self_token, [0.0, 0.0]))
            if self_token.size < self.obs_token_dim:
                self_token = np.pad(
                    self_token,
                    (0, self.obs_token_dim - self_token.size),
                )
            columns = [relative_gain[ue], load_ratio[ue], scale_token[ue], served[ue]]
            if self.obs_include_xy:
                columns.extend([ap_xy_token[ue, :, 0], ap_xy_token[ue, :, 1]])
            if self.fronthaul_queue:
                columns.append(queue_ratio[ue])
            if self.obs_ap_competition:
                columns.extend([
                    candidate_count_token[ue],
                    candidate_demand_token[ue],
                ])
            ap_tokens = np.column_stack(columns)
            neighbor_tokens = np.zeros(
                (self.obs_neighbor_ues, self.obs_token_dim),
                dtype=float,
            )
            if neighbor_order is not None:
                for (index, other) in enumerate(neighbor_order[ue]):
                    delta_xy = self.users_xy[other] - self.users_xy[ue]
                    relative_xy = np.clip(
                        delta_xy / max(0.5 * self.map_size, 1e-30),
                        -1.0,
                        1.0,
                    )
                    relative_distance = np.clip(
                        float(np.linalg.norm(delta_xy)) / self.map_size,
                        0.0,
                        1.0,
                    )
                    values = np.asarray([
                        relative_xy[0],
                        relative_xy[1],
                        relative_distance,
                        channel_correlation[ue, other],
                        candidate_overlap[ue, other],
                        rate_token[other],
                        qos_gap[other],
                        previous_mode_token[other],
                        previous_cluster_token[other],
                    ], dtype=float)
                    neighbor_tokens[index, :min(self.obs_token_dim, values.size)] = (
                        values[:self.obs_token_dim]
                    )
            observations.append(np.concatenate((
                self_token,
                neighbor_tokens.reshape(-1),
                ap_tokens.reshape(-1),
            )).astype(np.float32))
        return observations

    def _stack_observation_history(self, current_observations: List[np.ndarray]) -> List[np.ndarray]:
        current = np.asarray(current_observations, dtype=np.float32)
        if self._obs_history is None or self._obs_history.shape != (self.agent_num, self.obs_frames, self.obs_frame_dim):
            self._obs_history = np.zeros_like(current[:, None, :], shape=(self.agent_num, self.obs_frames, self.obs_frame_dim))
        if self.obs_frames == 1:
            self._obs_history[:, 0, :] = current
        else:
            self._obs_history[:, :-1, :] = self._obs_history[:, 1:, :]
            self._obs_history[:, -1, :] = current
        return [row.reshape(-1).copy() for row in self._obs_history]

    def step(self, actions: List[np.ndarray], collect_infos: bool=True):
        self._t += 1
        (mode_idx, cluster_size, action_idx) = self._decode_iicms_actions(actions)
        clusters = self._clusters_for_actions(action_idx)
        changed = self._update_cluster_changes(clusters)
        reported_cluster_changed = np.zeros_like(changed) if self.cluster_policy == 'static' else changed
        metrics = self._compute_metrics(mode_idx, cluster_size, clusters, advance_fronthaul_queue=True)
        rates = metrics['rate']
        mean_ue_se = float(np.mean(rates)) / self.bandwidth_hz
        system_sum_se = float(np.sum(rates)) / self.bandwidth_hz
        qos = self._qos_terms(rates)
        qos_penalty = float(qos['penalty'])
        # Shared team reward; zero QoS weight gives pure system-rate training.
        if self.reward_mode == 'mean_se':
            reward_values = np.full(self.agent_num, mean_ue_se - self.qos_penalty_weight * qos_penalty, dtype=float)
        elif self.reward_mode == 'mean_rate':
            reward_values = np.full(self.agent_num, float(np.mean(rates)) / 1000000.0 - self.qos_penalty_weight * qos_penalty, dtype=float)
        elif self.reward_mode == 'team_rate':
            reward_values = np.full(self.agent_num, float(np.sum(rates)) / 1000000.0 - self.qos_penalty_weight * qos_penalty, dtype=float)
        elif self.reward_mode == 'paper':
            reward_values = self._paper_reward(rates, mode_idx, clusters, metrics['offered_load'])
        elif self.reward_mode == 'team_sum':
            reward_values = np.full(self.agent_num, system_sum_se - self.qos_penalty_weight * qos_penalty, dtype=float)
        elif self.reward_mode == 'per_ue':
            reward_values = rates / self.bandwidth_hz - self.qos_penalty_weight * (qos['violation'].astype(float) + qos['normalized_shortfall'])
        else:
            raise ValueError(f'Unknown reward_mode: {self.reward_mode}')
        rewards = [np.array([value], dtype=np.float32) for value in reward_values]
        dones = [False] * self.agent_num
        interference_dbm = 10.0 * np.log10(np.maximum(metrics['interference'], 1e-12) / 0.001)
        primary = np.asarray([int(cluster[0]) for cluster in clusters], dtype=np.int32)
        self._previous_primary = primary
        self._previous_mode = mode_idx.copy()
        overload_ap = metrics['overloaded']
        total_fronthaul_arrival = float(np.sum(metrics['fronthaul_arrival_bits']))
        total_fronthaul_overflow = float(np.sum(metrics['fronthaul_current_drop_bits']))
        total_fronthaul_service_capacity = float(np.sum(np.maximum(self.backhaul_caps, 0.0) * self.slot_dt))
        total_fronthaul_service_used = float(np.sum(metrics['fronthaul_served_bits']) + np.sum(np.minimum(metrics['fronthaul_queue_before'], np.maximum(self.backhaul_caps, 0.0) * self.slot_dt)))
        self._last_step_summary = {'team_rate_mbps': float(np.mean(rates)) / 1000000.0, 'sum_rate_mbps': float(np.sum(rates)) / 1000000.0, 'minimum_ue_se_bps_hz': float(np.min(rates / self.bandwidth_hz)), 'se_floor_violation_rate': float(np.mean(rates / self.bandwidth_hz < self.min_ue_se_bps_hz)) if self.min_ue_se_bps_hz > 0.0 else 0.0, 'mean_ue_speed_mps': float(np.mean(self._speed)), 'mean_ue_se_bps_hz': mean_ue_se, 'system_sum_se_bps_hz': system_sum_se, 'objective_horizon': float(self.objective_horizon), 'qos_violation_rate': float(np.mean(qos['violation'])), 'qos_penalty': qos_penalty, 'min_rate_requirement_mbps': float(np.mean(self._rate_req_mbps)), 'interference_dbm': float(np.mean(interference_dbm)), 'backhaul_scale': float(np.mean(metrics['backhaul_scale'])), 'overload_ratio': float(np.mean(overload_ap)), 'fronthaul_queue_ratio': float(np.mean(metrics['fronthaul_queue_ratio'])), 'fronthaul_queue_active_ratio': float(np.mean(metrics['fronthaul_queue_bits'] > 1e-09)), 'fronthaul_drop_ratio': total_fronthaul_overflow / max(total_fronthaul_arrival, 1e-30), 'fronthaul_service_utilization': total_fronthaul_service_used / max(total_fronthaul_service_capacity, 1e-30), 'fronthaul_queue_delay_ms': float(np.mean(metrics['fronthaul_queue_delay_ms'])), 'cluster_change_rate': float(np.mean(reported_cluster_changed)), 'rho2': float(np.mean(self._rho2)), 'mode_jt': float(np.mean(mode_idx == 0)), 'mode_cb': float(np.mean(mode_idx == 1)), 'mode_nc': float(np.mean(mode_idx == 2)), 'cluster_size_mean': float(np.mean(cluster_size))}
        infos = []
        if collect_infos:
            for ue in range(self.agent_num):
                joint_index = int(action_idx[ue])
                primary_ap = int(clusters[ue][0])
                infos.append({'inst_rate': float(rates[ue]), 'spectral_efficiency_bps_hz': float(rates[ue]) / self.bandwidth_hz, 'se_floor_violation': float(rates[ue] / self.bandwidth_hz < self.min_ue_se_bps_hz) if self.min_ue_se_bps_hz > 0.0 else 0.0, 'inst_rate_pre': float(metrics['rate_pre'][ue]), 'inst_rate_backhaul': float(metrics['rate_bh'][ue]), 'interference_dbm': float(interference_dbm[ue]), 'sinr_clean': float(metrics['sinr_clean'][ue]), 'sinr_aged': float(metrics['sinr'][ue]), 'rate_requirement_mbps': float(self._rate_req_mbps[ue]), 'qos_violation': float(qos['violation'][ue]), 'qos_shortfall_mbps': float(qos['shortfall_mbps'][ue]), 'qos_penalty_term': float(qos['violation'][ue] + qos['normalized_shortfall'][ue]), 'reward': float(reward_values[ue]), 'backhaul_scale': float(metrics['backhaul_scale'][ue]), 'overload': float(metrics['backhaul_scale'][ue] < 0.999999), 'offered_load_bps': float(metrics['offered_load'][primary_ap]), 'fronthaul_queue_ratio': float(metrics['fronthaul_queue_ratio'][primary_ap]), 'fronthaul_drop_ratio': float(metrics['fronthaul_drop_ratio'][primary_ap]), 'fronthaul_queue_delay_ms': float(metrics['fronthaul_queue_delay_ms'][primary_ap]), 'rho2': float(self._rho2[ue]), 'mode': int(mode_idx[ue]), 'cluster_k': int(cluster_size[ue]), 'joint_action_index': joint_index, 'joint_action_name': self.action_names[joint_index], 'cluster_set': tuple((int(ap) for ap in clusters[ue])), 'cluster_subset_ranks': tuple((int(rank) for rank in self.joint_action_cluster_ranks[joint_index])), 'cluster_changed': float(reported_cluster_changed[ue]), 'position': self.users_xy[ue].copy(), 'speed': float(self._speed[ue])})
        self._last_inst_rate = rates.copy()
        self._last_interf_w = metrics['interference'].copy()
        self._advance_positions()
        self._update_channels()
        preview = self._compute_metrics(mode_idx, cluster_size, clusters, record_overload=False)
        current_obs = self._observe_from_state(preview['sinr'], preview['offered_load'], effective_rate_bps=preview['rate'], ap_scale=preview['ap_scale'], clusters=clusters)
        obs = self._stack_observation_history(current_obs)
        return (obs, rewards, dones, infos)

    def summary(self) -> Dict[str, object]:
        steps = max(1, self._t)
        return {'steps': self._t, 'mobility': self.mobility, 'ue_speed_mps': self.ue_speed, 'speed_mean_mps': float(np.mean(self._speed)), 'speed_min_mps': float(np.min(self._speed)), 'speed_max_mps': float(np.max(self._speed)), 'slot_dt_s': self.slot_dt, 'displacement_per_step_m': self.ue_speed * self.slot_dt, 'pathloss_model': self.pathloss_model, 'csi_aging': self.csi_aging, 'rho2_mean': float(np.mean(self._rho2)), 'size_options': list(self.size_options), 'cluster_action_mode': self.cluster_action_mode, 'cluster_policy': self.cluster_policy, 'obs_include_xy': self.obs_include_xy, 'obs_frames': self.obs_frames, 'obs_neighbor_ues': self.obs_neighbor_ues, 'obs_ap_competition': int(self.obs_ap_competition), 'obs_token_dim': self.obs_token_dim, 'obs_tokens_per_frame': self.obs_tokens_per_frame, 'obs_frame_dim': self.obs_frame_dim, 'joint_num_cluster_choices': self.joint_num_cluster_choices, 'reward_mode': self.reward_mode, 'objective_horizon': self.objective_horizon, 'objective_horizon_seconds': self.objective_horizon * self.slot_dt, 'reward_phi1': self.reward_phi1, 'reward_phi2': self.reward_phi2, 'min_rate_requirement_mbps': float(np.mean(self._rate_req_mbps)), 'min_ue_se_bps_hz': self.min_ue_se_bps_hz, 'qos_penalty_weight': self.qos_penalty_weight, 'backhaul_jt_model': self.backhaul_jt_model, 'fronthaul_queue': int(self.fronthaul_queue), 'fronthaul_buffer_s': self.fronthaul_buffer_s, 'ap_min_distance_m': self.ap_min_distance, 'cluster_changes_per_ue_step': float(self.cluster_changes.sum() / (steps * self.agent_num)), 'overload_events_per_ue_step': float(self.overload_events.sum() / (steps * self.agent_num)), 'per_ue_cluster_changes': self.cluster_changes.tolist()}

class IICMSMobileDiscreteActionEnv:

    def __init__(self):
        self.env = IICMSMobileEnvCore()
        self.obs_token_dim = int(getattr(self.env, 'obs_token_dim', 1))
        self.obs_frames = int(getattr(self.env, 'obs_frames', 1))
        self.obs_tokens_per_frame = int(getattr(self.env, 'obs_tokens_per_frame', self.env.obs_dim))
        self._refresh_and_build_spaces()

    def _refresh_and_build_spaces(self) -> None:
        self.num_agent = int(self.env.agent_num)
        self.signal_obs_dim = int(self.env.obs_dim)
        self.obs_token_dim = int(getattr(self.env, 'obs_token_dim', 1))
        self.obs_frames = int(getattr(self.env, 'obs_frames', 1))
        self.obs_tokens_per_frame = int(getattr(self.env, 'obs_tokens_per_frame', self.signal_obs_dim // max(1, self.obs_token_dim)))
        self.joint_num_modes = 3
        self.joint_num_sizes = int(self.env.joint_num_cluster_choices)
        self.joint_num_cluster_choices = int(self.env.joint_num_cluster_choices)
        self.joint_num_actions = len(self.env.joint_actions)
        self.joint_action_indices = list(self.env.joint_valid_flat_indices)
        self.joint_action_names = list(self.env.action_names)
        self.joint_cluster_choice_ranks = [list(ranks) for ranks in self.env.cluster_choice_ranks]
        self.joint_cluster_choice_sizes = list(self.env.cluster_choice_sizes)
        self.joint_action_cluster_choice = list(self.env.joint_action_cluster_choice)
        self.joint_num_ap_tokens = int(self.env.cluster_max)
        self.signal_action_dim = np.asarray(self.env.action_dim, dtype=int)
        share_obs_dim = self.signal_obs_dim * self.num_agent
        self.action_space = [spaces.MultiDiscrete(self.signal_action_dim) for _ in range(self.num_agent)]
        self.observation_space = [spaces.Box(low=-np.inf, high=np.inf, shape=(self.signal_obs_dim,), dtype=np.float32) for _ in range(self.num_agent)]
        self.share_observation_space = [spaces.Box(low=-np.inf, high=np.inf, shape=(share_obs_dim,), dtype=np.float32) for _ in range(self.num_agent)]

    def configure(self, all_args) -> None:
        sizes_raw = str(getattr(all_args, 'cluster_sizes', '3,4,5'))
        size_options = tuple((int(value.strip()) for value in sizes_raw.split(',') if value.strip()))
        kwargs = {'num_aps': getattr(all_args, 'num_aps', None), 'num_agents': getattr(all_args, 'num_agents', None), 'map_size': getattr(all_args, 'map_size', None), 'ap_min_distance': getattr(all_args, 'ap_min_distance', None), 'antennas_per_ap': getattr(all_args, 'antennas_per_ap', None), 'channel_file': getattr(all_args, 'channel_file', None), 'seed': getattr(all_args, 'seed', None), 'backhaul_cap_range': (getattr(all_args, 'backhaul_lo', 400000000.0), getattr(all_args, 'backhaul_hi', 500000000.0)), 'size_options': size_options, 'mobility': getattr(all_args, 'mobility', None), 'ue_speed': getattr(all_args, 'ue_speed', None), 'slot_dt': getattr(all_args, 'slot_dt', None), 'pathloss_model': getattr(all_args, 'pathloss_model', None), 'csi_aging': getattr(all_args, 'csi_aging', None), 'carrier_ghz': getattr(all_args, 'carrier_ghz', None), 'csi_block_slots': getattr(all_args, 'csi_block_slots', None), 'csi_pilot_slots': getattr(all_args, 'csi_pilot_slots', None), 'csi_slot_ms': getattr(all_args, 'csi_slot_ms', None), 'reward_mode': getattr(all_args, 'reward_mode', None), 'objective_horizon': getattr(all_args, 'objective_horizon', None), 'reward_phi1': getattr(all_args, 'reward_phi1', None), 'reward_phi2': getattr(all_args, 'reward_phi2', None), 'rate_req_lo_mbps': getattr(all_args, 'rate_req_lo_mbps', None), 'rate_req_hi_mbps': getattr(all_args, 'rate_req_hi_mbps', None), 'min_ue_se_bps_hz': getattr(all_args, 'min_ue_se_bps_hz', None), 'qos_penalty_weight': getattr(all_args, 'qos_penalty_weight', None), 'backhaul_jt_model': getattr(all_args, 'backhaul_jt_model', None), 'fronthaul_queue': getattr(all_args, 'fronthaul_queue', None), 'fronthaul_buffer_s': getattr(all_args, 'fronthaul_buffer_s', None), 'reset_ues': getattr(all_args, 'reset_ues', None), 'cluster_action_mode': getattr(all_args, 'cluster_action_mode', None), 'cluster_policy': getattr(all_args, 'cluster_policy', None), 'obs_include_xy': getattr(all_args, 'obs_include_xy', None), 'obs_frames': getattr(all_args, 'obs_frames', None), 'obs_neighbor_ues': getattr(all_args, 'obs_neighbor_ues', None), 'obs_ap_competition': getattr(all_args, 'obs_ap_competition', None)}
        self.env.configure(**{key: value for (key, value) in kwargs.items() if value is not None})
        self._refresh_and_build_spaces()

    def step(self, actions):
        (obs, rewards, dones, infos) = self.env.step(actions)
        return (np.stack(obs), np.stack(rewards), np.stack(dones), infos)

    def step_compact(self, actions):
        (obs, rewards, dones, _) = self.env.step(actions, collect_infos=False)
        return (np.stack(obs), np.stack(rewards), np.stack(dones), self.env._last_step_summary)

    def reset(self):
        return np.stack(self.env.reset())

    def seed(self, seed) -> None:
        self.env.seed(seed)

    def close(self) -> None:
        if hasattr(self.env, 'close'):
            self.env.close()

    def load_channel(self, path_or_dict) -> bool:
        ok = self.env.load_channel(path_or_dict)
        if ok:
            self._refresh_and_build_spaces()
        return ok

    def save_channel(self, path: str):
        return self.env.save_channel(path)
