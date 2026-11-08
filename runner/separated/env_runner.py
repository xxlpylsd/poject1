import time
import os
import csv
import json
import numpy as np
from itertools import chain
import torch
try:
    import scipy.io as sio
except Exception:
    sio = None

from utils.util import update_linear_schedule
from runner.separated.base_runner import Runner
from envs.env_wrappers import summarize_env_infos


def _t2n(x):
    return x.detach().cpu().numpy()


class EnvRunner(Runner):
    def __init__(self, config):
        super(EnvRunner, self).__init__(config)

    def run(self):
        self.warmup()
        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads
        reward_list = []
        curve_path = os.path.join(str(self.run_dir), "logs", "curve.csv")
        metrics_history = []
        last_progress = 0.0
        print(
            "[train-original-mappo] updates=%d rollout_envs=%d "
            "steps/update=%d target_env_steps=%d"
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
            if self.use_linear_lr_decay:
                for agent_id in range(self.num_agents):
                    self.trainer[agent_id].policy.lr_decay(episode, episodes)
            for step in range(self.episode_length):
                values,actions,action_log_probs,rnn_states,rnn_states_critic,actions_env = self.collect(step)
                if hasattr(self.envs, "step_compact"):
                    obs, rewards, dones, infos = (
                        self.envs.step_compact(actions_env)
                    )
                    step_metrics = (
                        infos
                        if isinstance(infos, dict)
                        else summarize_env_infos(infos)
                    )
                else:
                    obs, rewards, dones, infos = self.envs.step(
                        actions_env
                    )
                    step_metrics = summarize_env_infos(infos)
                episode_metrics.append(step_metrics)
                reward_list.append(rewards)
                data = (obs,rewards,dones,infos,values,actions,action_log_probs,rnn_states,rnn_states_critic)
                self.insert(data)

            # update
            self.compute()
            train_infos = self.train()

            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads

            if episode % self.save_interval == 0 or episode == episodes - 1:
                self.save()

            if episode_metrics:
                averaged_metrics = {
                    key: float(np.mean([
                        metric[key] for metric in episode_metrics
                    ]))
                    for key in episode_metrics[0]
                }
            else:
                averaged_metrics = {}
            curve_row = {
                "episode": int(episode + 1),
                "environment_steps": int(total_num_steps),
                **averaged_metrics,
            }
            write_header = not os.path.exists(curve_path)
            with open(
                curve_path, "a", newline="", encoding="utf-8-sig"
            ) as stream:
                writer = csv.DictWriter(
                    stream, fieldnames=list(curve_row.keys())
                )
                if write_header:
                    writer.writeheader()
                writer.writerow(curve_row)
            metrics_history.append(curve_row)
            with open(
                os.path.join(str(self.run_dir), "logs",
                             "metrics_history.json"),
                "w",
                encoding="utf-8",
            ) as stream:
                json.dump(
                    metrics_history, stream,
                    ensure_ascii=False, indent=2,
                )

            now = time.time()
            if (
                now - last_progress >= 5.0
                or episode == episodes - 1
            ):
                elapsed = max(now - start, 1e-9)
                fps = total_num_steps / elapsed
                remaining = max(
                    0, self.num_env_steps - total_num_steps
                )
                eta_s = int(remaining / max(fps, 1e-9))
                print(
                    "TRAIN-ORIG %d/%d | steps=%d/%d | FPS=%.1f | "
                    "ETA=%02d:%02d:%02d | rate=%.2f Mbps | "
                    "SE=%.2f | QoS=%.3f | JT/CB/NC=%.2f/%.2f/%.2f"
                    % (
                        episode + 1,
                        episodes,
                        total_num_steps,
                        self.num_env_steps,
                        fps,
                        eta_s // 3600,
                        (eta_s % 3600) // 60,
                        eta_s % 60,
                        averaged_metrics.get("sum_rate_mbps", 0.0),
                        averaged_metrics.get(
                            "system_sum_se_bps_hz", 0.0
                        ),
                        averaged_metrics.get(
                            "qos_violation_rate", 0.0
                        ),
                        averaged_metrics.get("mode_jt", 0.0),
                        averaged_metrics.get("mode_cb", 0.0),
                        averaged_metrics.get("mode_nc", 0.0),
                    ),
                    flush=True,
                )
                last_progress = now

            if episode % self.log_interval == 0:
                end = time.time()
                print(f"\n Scenario {self.all_args.scenario_name} Algo {self.algorithm_name} "
                      f"Exp {self.experiment_name} updates {episode}/{episodes}, "
                      f"total timesteps {total_num_steps}/{self.num_env_steps}, "
                      f"FPS {int(total_num_steps/(end-start))}.\n")

                # 统一记录平均 episode 回报（不依赖 env_name）
                for agent_id in range(self.num_agents):
                    train_infos[agent_id].update({
                        "average_episode_rewards": float(np.mean(self.buffer[agent_id].rewards) * self.episode_length)
                    })
                self.log_train(train_infos, total_num_steps)

            if episode % self.eval_interval == 0 and self.use_eval:
                self.eval(total_num_steps)

        # 保存到 run_dir
        if sio is not None:
            mat_path = os.path.join(str(self.run_dir), "reward_proposed.mat")
            sio.savemat(mat_path, {'reward_proposed': reward_list})
        else:
            np.save(os.path.join(str(self.run_dir), "reward_proposed.npy"),
                    np.asarray(reward_list))

    def warmup(self):
        # reset env
        obs = self.envs.reset()  # shape = [env_num, agent_num, obs_dim]

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)  # shape = [env_num, agent_num * obs_dim]

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))
            self.buffer[agent_id].share_obs[0] = share_obs.copy()
            self.buffer[agent_id].obs[0] = np.array(list(obs[:, agent_id])).copy()

    @torch.no_grad()
    def collect(self, step):
        values = []
        actions = []
        temp_actions_env = []
        action_log_probs = []
        rnn_states = []
        rnn_states_critic = []

        for agent_id in range(self.num_agents):             # 分别求所有智能体的trans
            self.trainer[agent_id].prep_rollout()           # 所有智能体的网络
            value, action, action_log_prob, rnn_state, rnn_state_critic = self.trainer[
                agent_id
            ].policy.get_actions(
                self.buffer[agent_id].share_obs[step],
                self.buffer[agent_id].obs[step],
                self.buffer[agent_id].rnn_states[step],
                self.buffer[agent_id].rnn_states_critic[step],
                self.buffer[agent_id].masks[step],
            )
            # [agents, envs, dim]
            values.append(_t2n(value))
            action = _t2n(action)
            # rearrange action
            if self.envs.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
                for i in range(self.envs.action_space[agent_id].nvec.size):
                    uc_action_env = np.eye(self.envs.action_space[agent_id].nvec[i])[action[:, i]]
                    if i == 0:
                        action_env = uc_action_env
                    else:
                        action_env = np.concatenate((action_env, uc_action_env), axis=1)
            elif self.envs.action_space[agent_id].__class__.__name__ == "Discrete":
                action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)      # 转为二进制的动作
            else:
                # TODO 这里改造成自己环境需要的形式即可
                # TODO Here, you can change the action_env to the form you need
                action_env = action
                # raise NotImplementedError

            actions.append(action)
            temp_actions_env.append(action_env)
            action_log_probs.append(_t2n(action_log_prob))
            rnn_states.append(_t2n(rnn_state))
            rnn_states_critic.append(_t2n(rnn_state_critic))

        # [envs, agents, dim]
        actions_env = []
        for i in range(self.n_rollout_threads):
            one_hot_action_env = []
            for temp_action_env in temp_actions_env:
                one_hot_action_env.append(temp_action_env[i])
            actions_env.append(one_hot_action_env)

        values = np.array(values).transpose(1, 0, 2)
        actions = np.array(actions).transpose(1, 0, 2)
        action_log_probs = np.array(action_log_probs).transpose(1, 0, 2)
        rnn_states = np.array(rnn_states).transpose(1, 0, 2, 3)
        rnn_states_critic = np.array(rnn_states_critic).transpose(1, 0, 2, 3)

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
            ((dones == True).sum(), self.recurrent_N, self.hidden_size),
            dtype=np.float32,
        )
        masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
        masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

        share_obs = []
        for o in obs:
            share_obs.append(list(chain(*o)))
        share_obs = np.array(share_obs)

        for agent_id in range(self.num_agents):
            if not self.use_centralized_V:
                share_obs = np.array(list(obs[:, agent_id]))

            self.buffer[agent_id].insert(
                share_obs,
                np.array(list(obs[:, agent_id])),
                rnn_states[:, agent_id],
                rnn_states_critic[:, agent_id],
                actions[:, agent_id],
                action_log_probs[:, agent_id],
                values[:, agent_id],
                rewards[:, agent_id],
                masks[:, agent_id],
            )

    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode_rewards = []
        eval_obs = self.eval_envs.reset()

        eval_rnn_states = np.zeros(
            (
                self.n_eval_rollout_threads,
                self.num_agents,
                self.recurrent_N,
                self.hidden_size,
            ),
            dtype=np.float32,
        )
        eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)

        for eval_step in range(self.episode_length):
            eval_temp_actions_env = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_action, eval_rnn_state = self.trainer[agent_id].policy.act(
                    np.array(list(eval_obs[:, agent_id])),
                    eval_rnn_states[:, agent_id],
                    eval_masks[:, agent_id],
                    deterministic=True,
                )

                eval_action = eval_action.detach().cpu().numpy()
                # rearrange action
                if self.eval_envs.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
                    space = self.eval_envs.action_space[agent_id]
                    # gym.spaces.MultiDiscrete: 用 nvec
                    if hasattr(space, "nvec"):
                        nvec = np.asarray(space.nvec, dtype=int)
                        for i in range(nvec.size):
                            n_i = int(nvec[i])
                            one = np.eye(n_i)[eval_action[:, i]]
                            if i == 0:
                                eval_action_env = one
                            else:
                                eval_action_env = np.concatenate((eval_action_env, one), axis=1)
                    else:
                        # 回退：自定义 MultiDiscrete（有 .high/.shape）
                        k = space.shape if isinstance(space.shape, int) else int(space.shape[0])
                        for i in range(k):
                            n_i = int(space.high[i] + 1)
                            one = np.eye(n_i)[eval_action[:, i]]
                            if i == 0:
                                eval_action_env = one
                            else:
                                eval_action_env = np.concatenate((eval_action_env, one), axis=1)

                elif self.eval_envs.action_space[agent_id].__class__.__name__ == "Discrete":
                    eval_action_env = np.squeeze(
                        np.eye(self.eval_envs.action_space[agent_id].n)[eval_action], 1
                    )
                else:
                    raise NotImplementedError

                eval_temp_actions_env.append(eval_action_env)
                eval_rnn_states[:, agent_id] = _t2n(eval_rnn_state)

            # [envs, agents, dim]
            eval_actions_env = []
            for i in range(self.n_eval_rollout_threads):
                eval_one_hot_action_env = []
                for eval_temp_action_env in eval_temp_actions_env:
                    eval_one_hot_action_env.append(eval_temp_action_env[i])
                eval_actions_env.append(eval_one_hot_action_env)

            # Obser reward and next obs
            eval_obs, eval_rewards, eval_dones, eval_infos = self.eval_envs.step(eval_actions_env)
            eval_episode_rewards.append(eval_rewards)

            eval_rnn_states[eval_dones == True] = np.zeros(
                ((eval_dones == True).sum(), self.recurrent_N, self.hidden_size),
                dtype=np.float32,
            )
            eval_masks = np.ones((self.n_eval_rollout_threads, self.num_agents, 1), dtype=np.float32)
            eval_masks[eval_dones == True] = np.zeros(((eval_dones == True).sum(), 1), dtype=np.float32)

        eval_episode_rewards = np.array(eval_episode_rewards)

        # 先按你原来的方式算每个 agent 的 episode 平均
        per_agent_avgs = []
        eval_train_infos = []
        for agent_id in range(self.num_agents):
            avg_i = float(np.mean(np.sum(eval_episode_rewards[:, :, agent_id], axis=0)))
            eval_train_infos.append({"eval_average_episode_rewards": avg_i})
            per_agent_avgs.append(avg_i)

        # 再计算“跨 agent 的平均值”，只打印这一个
        overall_avg = float(np.mean(per_agent_avgs))
        print(f"eval average episode rewards (mean over {self.num_agents} agents): {overall_avg:.3f}")

        # 原有日志照常写入（每个 agent 的曲线）
        self.log_train(eval_train_infos, total_num_steps)

        # 额外把“跨 agent 平均”写进 TensorBoard 方便看
        if hasattr(self, "writter"):
            self.writter.add_scalar("eval/avg_over_agents", overall_avg, total_num_steps)

    @torch.no_grad()
    def render(self):
        all_frames = []
        for episode in range(self.all_args.render_episodes):
            episode_rewards = []
            obs = self.envs.reset()
            if self.all_args.save_gifs:
                image = self.envs.render("rgb_array")[0][0]
                all_frames.append(image)

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

            for step in range(self.episode_length):
                calc_start = time.time()

                temp_actions_env = []
                for agent_id in range(self.num_agents):
                    if not self.use_centralized_V:
                        share_obs = np.array(list(obs[:, agent_id]))
                    self.trainer[agent_id].prep_rollout()
                    action, rnn_state = self.trainer[agent_id].policy.act(
                        np.array(list(obs[:, agent_id])),
                        rnn_states[:, agent_id],
                        masks[:, agent_id],
                        deterministic=True,
                    )

                    action = action.detach().cpu().numpy()
                    # rearrange action
                    if self.envs.action_space[agent_id].__class__.__name__ == "MultiDiscrete":
                        space = self.envs.action_space[agent_id]
                        if hasattr(space, "nvec"):
                            nvec = np.asarray(space.nvec, dtype=int)
                            for i in range(nvec.size):
                                n_i = int(nvec[i])
                                one = np.eye(n_i)[action[:, i]]
                                if i == 0:
                                    action_env = one
                                else:
                                    action_env = np.concatenate((action_env, one), axis=1)
                        else:
                            k = space.shape if isinstance(space.shape, int) else int(space.shape[0])
                            for i in range(k):
                                n_i = int(space.high[i] + 1)
                                one = np.eye(n_i)[action[:, i]]
                                if i == 0:
                                    action_env = one
                                else:
                                    action_env = np.concatenate((action_env, one), axis=1)

                    elif self.envs.action_space[agent_id].__class__.__name__ == "Discrete":
                        action_env = np.squeeze(np.eye(self.envs.action_space[agent_id].n)[action], 1)
                    else:
                        raise NotImplementedError

                    temp_actions_env.append(action_env)
                    rnn_states[:, agent_id] = _t2n(rnn_state)

                # [envs, agents, dim]
                actions_env = []
                for i in range(self.n_rollout_threads):
                    one_hot_action_env = []
                    for temp_action_env in temp_actions_env:
                        one_hot_action_env.append(temp_action_env[i])
                    actions_env.append(one_hot_action_env)

                # Obser reward and next obs
                obs, rewards, dones, infos = self.envs.step(actions_env)
                episode_rewards.append(rewards)

                rnn_states[dones == True] = np.zeros(
                    ((dones == True).sum(), self.recurrent_N, self.hidden_size),
                    dtype=np.float32,
                )
                masks = np.ones((self.n_rollout_threads, self.num_agents, 1), dtype=np.float32)
                masks[dones == True] = np.zeros(((dones == True).sum(), 1), dtype=np.float32)

                if self.all_args.save_gifs:
                    image = self.envs.render("rgb_array")[0][0]
                    all_frames.append(image)
                    calc_end = time.time()
                    elapsed = calc_end - calc_start
                    if elapsed < self.all_args.ifi:
                        time.sleep(self.all_args.ifi - elapsed)

            episode_rewards = np.array(episode_rewards)
            for agent_id in range(self.num_agents):
                average_episode_rewards = np.mean(np.sum(episode_rewards[:, :, agent_id], axis=0))
                print("eval average episode rewards of agent%i: " % agent_id + str(average_episode_rewards))

        if self.all_args.save_gifs and getattr(self, "imageio", None) is not None:
            # 兜底确保目录存在
            if not os.path.exists(self.gif_dir):
                os.makedirs(self.gif_dir, exist_ok=True)
            self.imageio.mimsave(
                os.path.join(self.gif_dir, "render.gif"),
                all_frames,
                duration=self.all_args.ifi,
            )

