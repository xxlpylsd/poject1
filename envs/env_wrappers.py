import multiprocessing as mp
import traceback
import numpy as np

def _worker(remote, parent_remote, env_fn):
    parent_remote.close()
    env = env_fn()
    metadata = {'obs_token_dim': int(getattr(env, 'obs_token_dim', 1)), 'obs_dim': int(getattr(env, 'signal_obs_dim', 1)), 'obs_frames': int(getattr(env, 'obs_frames', 1)), 'obs_tokens_per_frame': int(getattr(env, 'obs_tokens_per_frame', 1)), 'joint_num_modes': int(getattr(env, 'joint_num_modes', 3)), 'joint_num_sizes': int(getattr(env, 'joint_num_sizes', 0)), 'joint_num_cluster_choices': int(getattr(env, 'joint_num_cluster_choices', 0)), 'joint_num_actions': int(getattr(env, 'joint_num_actions', 0)), 'joint_action_indices': list(getattr(env, 'joint_action_indices', [])), 'joint_action_names': list(getattr(env, 'joint_action_names', [])), 'joint_cluster_choice_ranks': list(getattr(env, 'joint_cluster_choice_ranks', [])), 'joint_cluster_choice_sizes': list(getattr(env, 'joint_cluster_choice_sizes', [])), 'joint_action_cluster_choice': list(getattr(env, 'joint_action_cluster_choice', [])), 'joint_num_ap_tokens': int(getattr(env, 'joint_num_ap_tokens', 0))}
    remote.send(('ready', (env.observation_space, env.share_observation_space, env.action_space, metadata)))
    try:
        while True:
            (command, data) = remote.recv()
            if command == 'step':
                result = env.step(data)
            elif command == 'step_compact':
                if hasattr(env, 'step_compact'):
                    result = env.step_compact(data)
                else:
                    result = env.step(data)
            elif command == 'reset':
                result = env.reset()
            elif command == 'close':
                env.close()
                remote.close()
                return
            else:
                raise ValueError(f'Unknown vector-env command: {command}')
            remote.send(('ok', result))
    except EOFError:
        return
    except Exception:
        try:
            remote.send(('error', traceback.format_exc()))
        except Exception:
            pass
    finally:
        remote.close()

def summarize_env_infos(infos):
    flattened = [item for env_infos in infos for item in env_infos]
    return {'team_rate_mbps': float(np.mean([item['inst_rate'] for item in flattened])) / 1000000.0, 'sum_rate_mbps': float(np.sum([item['inst_rate'] for item in flattened])) / 1000000.0, 'mean_ue_se_bps_hz': float(np.mean([item['spectral_efficiency_bps_hz'] for item in flattened])), 'system_sum_se_bps_hz': float(np.sum([item['spectral_efficiency_bps_hz'] for item in flattened])), 'interference_dbm': float(np.mean([item['interference_dbm'] for item in flattened])), 'backhaul_scale': float(np.mean([item['backhaul_scale'] for item in flattened])), 'overload_ratio': float(np.mean([item['overload'] for item in flattened])), 'qos_violation_rate': float(np.mean([item['qos_violation'] for item in flattened])), 'qos_penalty_term': float(np.mean([item['qos_penalty_term'] for item in flattened])), 'rho2': float(np.mean([item['rho2'] for item in flattened])), 'cluster_change_rate': float(np.mean([item['cluster_changed'] for item in flattened])), 'mode_jt': float(np.mean([item['mode'] == 0 for item in flattened])), 'mode_cb': float(np.mean([item['mode'] == 1 for item in flattened])), 'mode_nc': float(np.mean([item['mode'] == 2 for item in flattened])), 'cluster_size_mean': float(np.mean([item['cluster_k'] for item in flattened]))}

def aggregate_step_summaries(summaries):
    keys = summaries[0].keys()
    return {key: float(np.mean([summary[key] for summary in summaries])) for key in keys}

class DummyVecEnv:

    def __init__(self, env_fns):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        self.num_envs = len(env_fns)
        self.observation_space = env.observation_space
        self.share_observation_space = env.share_observation_space
        self.action_space = env.action_space
        self.actions = None

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        self.actions = actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        (obs, rews, dones, infos) = map(np.array, zip(*results))
        for (i, done) in enumerate(dones):
            if 'bool' in done.__class__.__name__:
                if done:
                    obs[i] = self.envs[i].reset()
            elif np.all(done):
                obs[i] = self.envs[i].reset()
        self.actions = None
        return (obs, rews, dones, infos)

    def step_compact(self, actions):
        results = []
        for (action, env) in zip(actions, self.envs):
            if hasattr(env, 'step_compact'):
                results.append(env.step_compact(action))
            else:
                results.append(env.step(action))
        (obs, rews, dones, details) = map(np.array, zip(*results))
        summaries = []
        for (i, detail) in enumerate(details):
            if isinstance(detail, dict):
                summaries.append(detail)
            else:
                summaries.append(summarize_env_infos([detail]))
            done = dones[i]
            finished = bool(done) if 'bool' in done.__class__.__name__ else bool(np.all(done))
            if finished:
                obs[i] = self.envs[i].reset()
        return (obs, rews, dones, aggregate_step_summaries(summaries))

    def reset(self):
        obs = [env.reset() for env in self.envs]
        return np.array(obs)

    def close(self):
        for env in self.envs:
            env.close()

    def render(self, mode='human'):
        if mode == 'rgb_array':
            return np.array([env.render(mode=mode) for env in self.envs])
        elif mode == 'human':
            for env in self.envs:
                env.render(mode=mode)
        else:
            raise NotImplementedError

class SubprocVecEnv:

    def __init__(self, env_fns, start_method='fork'):
        self.num_envs = len(env_fns)
        context = mp.get_context(start_method)
        self.parent_remotes = []
        self.remotes = []
        self.processes = []
        self.actions = None
        for env_fn in env_fns:
            (parent_remote, remote) = context.Pipe()
            process = context.Process(target=_worker, args=(remote, parent_remote, env_fn), daemon=True)
            process.start()
            remote.close()
            self.parent_remotes.append(parent_remote)
            self.remotes.append(remote)
            self.processes.append(process)
        ready = [self._receive(remote) for remote in self.parent_remotes]
        (self.observation_space, self.share_observation_space, self.action_space, self.worker_metadata) = ready[0]

    @staticmethod
    def _receive(remote):
        (status, payload) = remote.recv()
        if status == 'error':
            raise RuntimeError(f'Vector-environment worker failed:\n{payload}')
        return payload

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def step_async(self, actions):
        self.actions = actions
        for (remote, action) in zip(self.parent_remotes, actions):
            remote.send(('step', action))

    def step_wait(self):
        results = [self._receive(remote) for remote in self.parent_remotes]
        (obs, rews, dones, infos) = map(np.array, zip(*results))
        for (i, done) in enumerate(dones):
            finished = bool(done) if 'bool' in done.__class__.__name__ else bool(np.all(done))
            if finished:
                self.parent_remotes[i].send(('reset', None))
                obs[i] = self._receive(self.parent_remotes[i])
        self.actions = None
        return (obs, rews, dones, infos)

    def step_compact(self, actions):
        for (remote, action) in zip(self.parent_remotes, actions):
            remote.send(('step_compact', action))
        results = [self._receive(remote) for remote in self.parent_remotes]
        (obs, rews, dones, details) = map(np.array, zip(*results))
        summaries = []
        for (i, detail) in enumerate(details):
            if isinstance(detail, dict):
                summaries.append(detail)
            else:
                summaries.append(summarize_env_infos([detail]))
            done = dones[i]
            finished = bool(done) if 'bool' in done.__class__.__name__ else bool(np.all(done))
            if finished:
                self.parent_remotes[i].send(('reset', None))
                obs[i] = self._receive(self.parent_remotes[i])
        return (obs, rews, dones, aggregate_step_summaries(summaries))

    def reset(self):
        for remote in self.parent_remotes:
            remote.send(('reset', None))
        return np.array([self._receive(remote) for remote in self.parent_remotes])

    def close(self):
        for remote in self.parent_remotes:
            try:
                remote.send(('close', None))
            except (BrokenPipeError, EOFError):
                pass
        for process in self.processes:
            process.join(timeout=3)
        for process in self.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=1)
        for remote in self.parent_remotes:
            remote.close()

    def render(self, mode='human'):
        raise NotImplementedError("Rendering requires vec_env='dummy'.")

def make_vec_env(env_fns, vec_env='auto'):
    kind = str(vec_env).lower()
    if kind not in {'auto', 'dummy', 'subproc'}:
        raise ValueError(f'Unknown vec_env mode: {vec_env}')
    can_fork = 'fork' in mp.get_all_start_methods()
    if kind == 'subproc' and (not can_fork):
        raise RuntimeError("vec_env='subproc' requires fork support (Linux).")
    if kind == 'dummy' or len(env_fns) == 1 or (kind == 'auto' and (not can_fork)):
        print(f'[vec_env] mode=dummy workers={len(env_fns)}', flush=True)
        return DummyVecEnv(env_fns)
    print(f'[vec_env] mode=subproc workers={len(env_fns)}', flush=True)
    return SubprocVecEnv(env_fns)
