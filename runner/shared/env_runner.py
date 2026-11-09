"""
# @Time    : 2021/7/1 7:15 下午
# @Author  : hezhiqiang01
# @Email   : hezhiqiang01@baidu.com
# @File    : env_runner.py
"""

import time
import csv
import json
import os
import sys
import numpy as np
import torch
try:
    import scipy.io as sio
except Exception:
    sio = None
from runner.shared.base_runner import Runner
from envs.env_wrappers import summarize_env_infos
from utils.policy_evaluation import (
    SharedPolicyActor,
    evaluate_episodes,
)

# import imageio


def _t2n(x):
    return x.detach().cpu().numpy()


_OBJECTIVE_METRIC_SPECS = {
    "mean_se": ("MeanSE", "mean_ue_se_bps_hz", "bps/Hz"),
    "mean_rate": ("MeanRate", "team_rate_mbps", "Mbps"),
    "team_rate": ("SysRate", "sum_rate_mbps", "Mbps"),
    "team_sum": ("SysSE", "system_sum_se_bps_hz", "bps/Hz"),
    "paper": ("SysSE", "system_sum_se_bps_hz", "bps/Hz"),
    "per_ue": ("MeanSE", "mean_ue_se_bps_hz", "bps/Hz"),
}


def _objective_metric(reward_mode, metrics):
    """Return the metric that directly reflects the configured reward."""
    label, key, unit = _OBJECTIVE_METRIC_SPECS.get(
        str(reward_mode),
        _OBJECTIVE_METRIC_SPECS["mean_se"],
    )
    try:
        value = float(metrics.get(key, 0.0))
    except (TypeError, ValueError):
        value = 0.0
    return label, value, unit


class _TrainingProgress:
    """Compact terminal and log-friendly training progress display."""

    def __init__(
        self,
        total_episodes,
        total_env_steps,
        reward_mode="mean_se",
        refresh_s=1.0,
    ):
        self.total_episodes = max(1, int(total_episodes))
        self.total_env_steps = max(1, int(total_env_steps))
        self.reward_mode = str(reward_mode)
        self.refresh_s = float(refresh_s)
        self.is_tty = bool(getattr(sys.stdout, "isatty", lambda: False)())
        self.last_update = 0.0

    @staticmethod
    def _duration(seconds):
        seconds = max(0, int(seconds))
        hours, remainder = divmod(seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    def update(
        self,
        episode_index,
        env_steps,
        elapsed_s,
        metrics,
        average_reward,
        force=False,
    ):
        now = time.time()
        if not force and now - self.last_update < self.refresh_s:
            return
        self.last_update = now

        completed = min(episode_index + 1, self.total_episodes)
        fraction = min(1.0, env_steps / float(self.total_env_steps))
        width = 24
        filled = int(width * fraction)
        bar = "#" * filled + "-" * (width - filled)
        fps = env_steps / max(elapsed_s, 1e-9)
        remaining = max(0.0, self.total_env_steps - env_steps)
        eta = remaining / max(fps, 1e-9)
        objective_label, objective_value, objective_unit = (
            _objective_metric(self.reward_mode, metrics)
        )
        line = (
            f"TRAIN {completed}/{self.total_episodes} [{bar}] "
            f"{100.0 * fraction:6.2f}% | "
            f"steps={env_steps}/{self.total_env_steps} | "
            f"FPS={fps:8.1f} | ETA={self._duration(eta)} | "
            f"reward={average_reward:.4g} | "
            f"objective={self.reward_mode} | "
            f"{objective_label}={objective_value:.2f}{objective_unit} | "
            f"QoS={metrics.get('qos_violation_rate', 0.0):.3f} | "
            f"gate={metrics.get('joint_gate_mean', 0.5):.3f} | "
            f"overload={metrics.get('overload_ratio', 0.0):.3f} | "
            f"L={metrics.get('cluster_size_mean', 0.0):.2f} | "
            f"JT/CB/NC={metrics.get('mode_jt', 0.0):.2f}/"
            f"{metrics.get('mode_cb', 0.0):.2f}/"
            f"{metrics.get('mode_nc', 0.0):.2f}"
        )
        if self.is_tty:
            print("\r" + line, end="", flush=True)
        else:
            print(line, flush=True)

    def finish(self):
        if self.is_tty:
            print(flush=True)


class EnvRunner(Runner):
    """Runner class to perform training, evaluation. and data collection for the MPEs. See parent class for details."""

    def __init__(self, config):
        super(EnvRunner, self).__init__(config)
        self.curve_path = os.path.join(self.log_dir, "curve.csv")
        self.metrics_history = []

    @staticmethod
    def _summarize_infos(infos):
        return summarize_env_infos(infos)

    def _append_curve(self, episode, total_steps, metrics):
        row = {
            "episode": int(episode + 1),
            "environment_steps": int(total_steps),
            **{key: float(value) for key, value in metrics.items()},
        }
        write_header = not os.path.exists(self.curve_path)
        with open(self.curve_path, "a", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(row.keys()))
            if write_header:
                writer.writeheader()
            writer.writerow(row)
        self.metrics_history.append(row)
        with open(
            os.path.join(self.log_dir, "metrics_history.json"),
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(self.metrics_history, f, ensure_ascii=False, indent=2)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        progress = _TrainingProgress(
            episodes,
            self.num_env_steps,
            reward_mode=self.all_args.reward_mode,
        )
        reward_list = []
        print(
            "[train] updates=%d, rollout_envs=%d, steps/update=%d, "
            "target_env_steps=%d"
            % (
                episodes,
                self.n_rollout_threads,
                self.episode_length * self.n_rollout_threads,
                self.num_env_steps,
            ),
            flush=True,
        )

        for episode in range(episodes):
            episode_metrics = []
            update_start = time.perf_counter()
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            for step in range(self.episode_length):
                # Sample actions
                (
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                    actions_env,            # 二进制动作
                ) = self.collect(step)      # 得到一次的动作采样值

                action_head = getattr(
                    self.trainer.policy.actor, "act", None
                )
                last_gate = getattr(action_head, "last_gate", None)
                gate_metrics = {}
                if last_gate is not None:
                    gate_metrics = {
                        "joint_gate_mean": float(
                            last_gate.detach().mean().cpu().item()
                        ),
                        "joint_gate_std": float(
                            last_gate.detach().std().cpu().item()
                        ),
                    }

                # Obser reward and next obs 动作采样值对应的观测
                if hasattr(self.envs, "step_compact"):
                    obs, rewards, dones, step_metrics = (
                        self.envs.step_compact(actions_env)
                    )
                    infos = step_metrics
                else:
                    obs, rewards, dones, infos = self.envs.step(
                        actions_env
                    )
                    step_metrics = self._summarize_infos(infos)
                step_metrics.update(gate_metrics)
                episode_metrics.append(step_metrics)
                reward_list.append(rewards)
                data = (
                    obs,
                    rewards,
                    dones,
                    infos,
                    values,
                    actions,
                    action_log_probs,
                    rnn_states,
                    rnn_states_critic,
                )

                # insert data into buffer
                self.insert(data)

            rollout_s = time.perf_counter() - update_start
            # compute return and update network
            learn_start = time.perf_counter()
            self.compute()
            train_infos = self.train()
            learn_s = time.perf_counter() - learn_start

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            # save model
            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save(total_num_steps)

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print(
                    "\n Scenario {} Algo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n".format(
                        self.all_args.scenario_name,
                        self.algorithm_name,
                        self.experiment_name,
                        episode,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        int(total_num_steps / (end - start)),
                    )
                )

                # if self.env_name == "MPE":
                #     env_infos = {}
                #     for agent_id in range(self.num_agents):
                #         idv_rews = []
                #         for info in infos:
                #             if 'individual_reward' in info[agent_id].keys():
                #                 idv_rews.append(info[agent_id]['individual_reward'])
                #         agent_k = 'agent%i/individual_rewards' % agent_id
                #         env_infos[agent_k] = idv_rews

                train_infos["average_episode_rewards"] = np.mean(self.buffer.rewards) * self.episode_length
                env_metrics = {
                    key: float(np.mean([m[key] for m in episode_metrics]))
                    for key in episode_metrics[0]
                }
                self._append_curve(episode, total_num_steps, env_metrics)
                print("average episode rewards is {}".format(train_infos["average_episode_rewards"]))
                objective_label, objective_value, objective_unit = (
                    _objective_metric(self.all_args.reward_mode, env_metrics)
                )
                print(
                    "train objective | "
                    f"reward_mode={self.all_args.reward_mode} | "
                    f"{objective_label}={objective_value:.2f}{objective_unit} | "
                    "QoS="
                    f"{env_metrics.get('qos_violation_rate', 0.0):.3f} | "
                    "overload="
                    f"{env_metrics.get('overload_ratio', 0.0):.3f} | "
                    "cluster_change="
                    f"{env_metrics.get('cluster_change_rate', 0.0):.3f}",
                    flush=True,
                )
                self.log_train(train_infos, total_num_steps)
                # self.log_env(env_infos, total_num_steps)

            env_metrics = {
                key: float(np.mean([m[key] for m in episode_metrics]))
                for key in episode_metrics[0]
            }
            progress.update(
                episode,
                total_num_steps,
                time.time() - start,
                env_metrics,
                float(np.mean(self.buffer.rewards) * self.episode_length),
                force=episode == episodes - 1,
            )
            if episode % self.log_interval == 0:
                print(
                    "timing rollout=%.3fs learn=%.3fs"
                    % (rollout_s, learn_s),
                    flush=True,
                )

            # fixed-seed deterministic validation
            if self.should_validate(
                total_num_steps,
                final=episode == episodes - 1,
            ):
                self.eval(total_num_steps)

        progress.finish()
        if sio is not None:
            sio.savemat(str(self.run_dir / 'reward_proposed.mat'),
                        {'reward_proposed': reward_list})
        else:
            np.save(self.run_dir / 'reward_proposed.npy', np.asarray(reward_list))

    def warmup(self):
        # reset env
        obs = self.envs.reset()  # shape = [env_num, agent_num, obs_dim]

        # replay buffer
        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)  # shape = [env_num, agent_num * obs_dim]共享的观测空间，相乘
            share_obs = np.expand_dims(share_obs, 1)
            if not self.critic_dedup:
                share_obs = share_obs.repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.share_obs[0] = share_obs.copy()  # share的观测和独立的观测分别存到self.buffer里
        self.buffer.obs[0] = obs.copy()

    @torch.no_grad()
    def collect(self, step):
        self.trainer.prep_rollout()
        critic_rnn_states = self.buffer.rnn_states_critic[step]
        if self.critic_dedup:
            critic_rnn_states = critic_rnn_states[:, :1]
        (
            value,
            action,
            action_log_prob,
            rnn_states,
            rnn_states_critic,
        ) = self.trainer.policy.get_actions(
            np.concatenate(self.buffer.share_obs[step]),
            np.concatenate(self.buffer.obs[step]),
            np.concatenate(self.buffer.rnn_states[step]),
            np.concatenate(critic_rnn_states),
            np.concatenate(self.buffer.masks[step]),
        )
        # [self.envs, agents, dim]
        # buffer中拿到的数据进行后处理,分为5个线程————对应5个环境
        values = np.array(np.split(_t2n(value), self.n_rollout_threads))  # [env_num, agent_num, 1]
        actions = np.array(np.split(_t2n(action), self.n_rollout_threads))  # [env_num, agent_num, action_dim]
        action_log_probs = np.array(
            np.split(_t2n(action_log_prob), self.n_rollout_threads)
        )  # [env_num, agent_num, 1]
        rnn_states = np.array(
            np.split(_t2n(rnn_states), self.n_rollout_threads)
        )  # [env_num, agent_num, 1, hidden_size]
        rnn_states_critic = np.array(
            np.split(_t2n(rnn_states_critic), self.n_rollout_threads)
        )  # [env_num, agent_num, 1, hidden_size]
        if self.critic_dedup:
            values = np.repeat(values, self.num_agents, axis=1)
            rnn_states_critic = np.repeat(
                rnn_states_critic, self.num_agents, axis=1
            )
        # rearrange action
        if self.envs.action_space[0].__class__.__name__ == "MultiDiscrete":
            nvec = self.envs.action_space[0].nvec
            for i in range(len(nvec)):
                uc_actions_env = np.eye(int(nvec[i]))[actions[:, :, i]]
                if i == 0:
                    actions_env = uc_actions_env
                else:
                    actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
        elif self.envs.action_space[0].__class__.__name__ == "Discrete":
            # actions  --> actions_env : shape:[10, 1] --> [5, 2, 5]
            actions_env = np.squeeze(np.eye(self.envs.action_space[0].n)[actions], 2)
        else:
            # TODO 这里改造成自己环境需要的形式即可
            # TODO Here, you can change the shape of actions_env to fit your environment
            actions_env = actions
            # raise NotImplementedError

        return (
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
            actions_env,
        )

    def insert(self, data):
        (
            obs,
            rewards,
            dones,
            infos,
            values,
            actions,
            action_log_probs,
            rnn_states,
            rnn_states_critic,
        ) = data

        rnn_states[dones == True] = np.zeros(
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        rnn_states_critic[dones == True] = np.zeros(
            ((dones == True).sum(), *self.buffer.rnn_states_critic.shape[3:]),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        if self.use_centralized_V:
            share_obs = obs.reshape(self.n_rollout_threads, -1)
            share_obs = np.expand_dims(share_obs, 1)
            if not self.critic_dedup:
                share_obs = share_obs.repeat(self.num_agents, axis=1)
        else:
            share_obs = obs

        self.buffer.insert(
            share_obs,
            obs,
            rnn_states,
            rnn_states_critic,
            actions,
            action_log_probs,
            values,
            rewards,
            masks,
        )

    def eval(self, total_num_steps):
        if self.eval_envs is None or not self.eval_envs.envs:
            raise RuntimeError(
                "Training validation requires an evaluation environment"
            )
        self.trainer.prep_rollout()
        policy = self.trainer.policy
        policy.actor.eval()
        policy.critic.eval()
        eval_env = self.eval_envs.envs[0]
        seeds = [
            self.validation_seed_start + index
            for index in range(self.validation_episodes)
        ]
        metrics = evaluate_episodes(
            eval_env,
            self.all_args,
            seeds,
            self.validation_horizon,
            SharedPolicyActor(
                policy, self.device, self.num_agents
            ),
        )
        return self.log_validation(metrics, total_num_steps)

    @torch.no_grad()
    def render(self):
        """Visualize the env."""
        envs = self.envs

        all_frames = []
        for episode in range(self.all_args.render_episodes):
            obs = envs.reset()
            if self.all_args.save_gifs:
                image = envs.render("rgb_array")[0][0]
                all_frames.append(image)
            else:
                envs.render("human")

            rnn_states = np.zeros(
                (
                    self.n_rollout_threads,
                    self.num_agents,
                    self.recurrent_N,
                    self.hidden_size,
                ),
                dtype=np.float32,
            )
            masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)

            episode_rewards = []

            for step in range(self.episode_length):
                calc_start = time.time()

                self.trainer.prep_rollout()
                action, rnn_states = self.trainer.policy.act(
                    np.concatenate(obs),
                    np.concatenate(rnn_states),
                    np.concatenate(masks),
                    deterministic=True,
                )
                actions = np.array(np.split(_t2n(action), self.n_rollout_threads))
                rnn_states = np.array(np.split(_t2n(rnn_states), self.n_rollout_threads))

                if envs.action_space[0].__class__.__name__ == "MultiDiscrete":
                    nvec = envs.action_space[0].nvec
                    for i in range(len(nvec)):
                        uc_actions_env = np.eye(int(nvec[i]))[actions[:, :, i]]
                        if i == 0:
                            actions_env = uc_actions_env
                        else:
                            actions_env = np.concatenate((actions_env, uc_actions_env), axis=2)
                elif envs.action_space[0].__class__.__name__ == "Discrete":
                    actions_env = np.squeeze(np.eye(envs.action_space[0].n)[actions], 2)
                else:
                    raise NotImplementedError

                # Obser reward and next obs
                obs, rewards, dones, infos = envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = envs.render("rgb_array")[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)
                else:
                    envs.render("human")

            print("average episode rewards is: " + str(np.mean(np.sum(np.array(episode_rewards), axis=0))))

        # if self.all_args.save_gifs:
        #     imageio.mimsave(str(self.gif_dir) + '/render.gif', all_frames, duration=self.all_args.ifi)
