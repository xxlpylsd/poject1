
import os
import numpy as np
from itertools import chain
import torch
from tensorboardX import SummaryWriter
import csv, time
from utils.separated_buffer import SeparatedReplayBuffer
from utils.util import update_linear_schedule
from utils.policy_evaluation import ValidationRecorder


def _t2n(x):
    return x.detach().cpu().numpy()


class Runner(object):
    def __init__(self, config):
        self.all_args = config["all_args"]
        self.envs = config["envs"]
        self.eval_envs = config["eval_envs"]
        self.device = config["device"]
        self.num_agents = config["num_agents"]
        self.run_dir = config["run_dir"]  # 用 Path 传进来的

        # parameters
        self.env_name = self.all_args.env_name
        self.algorithm_name = self.all_args.algorithm_name
        self.experiment_name = self.all_args.experiment_name
        self.use_centralized_V = self.all_args.use_centralized_V
        self.use_obs_instead_of_state = self.all_args.use_obs_instead_of_state
        self.num_env_steps = self.all_args.num_env_steps
        self.episode_length = self.all_args.episode_length
        self.n_rollout_threads = self.all_args.n_rollout_threads
        self.n_eval_rollout_threads = self.all_args.n_eval_rollout_threads
        self.use_linear_lr_decay = self.all_args.use_linear_lr_decay
        self.hidden_size = self.all_args.hidden_size
        self.use_render = self.all_args.use_render
        self.recurrent_N = self.all_args.recurrent_N

        # interval
        self.save_interval = self.all_args.save_interval
        self.use_eval = self.all_args.use_eval
        self.eval_interval = self.all_args.eval_interval
        self.validation_interval_steps = int(
            self.all_args.validation_interval_steps
        )
        self.validation_episodes = int(
            self.all_args.validation_episodes
        )
        self.validation_seed_start = int(
            self.all_args.validation_seed_start
        )
        self.validation_horizon = int(
            self.all_args.validation_horizon
        ) or int(self.episode_length)
        self.log_interval = self.all_args.log_interval

        # dir
        self.model_dir = self.all_args.model_dir

        # ---- 日志/模型目录：始终初始化一次（无论是否渲染）----
        self.log_dir = str(self.run_dir / "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.writter = SummaryWriter(self.log_dir)

        self.save_dir = str(self.run_dir / "models")
        os.makedirs(self.save_dir, exist_ok=True)
        self.next_validation_step = self.validation_interval_steps
        self.validation_recorder = ValidationRecorder(
            self.run_dir,
            float(
                self.all_args.validation_plateau_relative_gain
            ),
            int(self.all_args.validation_window),
        )

        # CSV 日志（GUI 轮询用）
        self._csv_path = os.path.join(self.log_dir, "progress.csv")
        if not os.path.exists(self._csv_path):
            with open(self._csv_path, "w", newline="", encoding="utf-8") as f:
                csv.writer(f).writerow(["time", "total_steps", "episode", "agent", "key", "value"])

        # ---- GIF/渲染相关：仅在需要时准备 imageio 与 gifs 目录 ----
        self.imageio = None
        if self.use_render or getattr(self.all_args, "save_gifs", False):
            try:
                import imageio
                self.imageio = imageio
            except Exception:
                self.imageio = None
            self.gif_dir = str(self.run_dir / "gifs")
            os.makedirs(self.gif_dir, exist_ok=True)

        from algorithms.algorithm.r_mappo import RMAPPO as TrainAlgo
        from algorithms.algorithm.rMAPPOPolicy import RMAPPOPolicy as Policy

        self.policy = []
        for agent_id in range(self.num_agents):
            share_observation_space = (
                self.envs.share_observation_space[agent_id]
                if self.use_centralized_V
                else self.envs.observation_space[agent_id]
            )
            po = Policy(
                self.all_args,
                self.envs.observation_space[agent_id],
                share_observation_space,
                self.envs.action_space[agent_id],
                device=self.device,
            )
            self.policy.append(po)
        print(
            "[model-original-mappo] independent_agents=%d "
            "actor=%s(%d params/agent) critic=%s(%d params/agent)"
            % (
                self.num_agents,
                self.policy[0].actor.backbone_name,
                self.policy[0].actor.num_params(),
                self.policy[0].critic.backbone_name,
                self.policy[0].critic.num_params(),
            ),
            flush=True,
        )

        if self.model_dir is not None:
            self.restore()

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            tr = TrainAlgo(self.all_args, self.policy[agent_id], device=self.device)
            share_observation_space = (
                self.envs.share_observation_space[agent_id]
                if self.use_centralized_V
                else self.envs.observation_space[agent_id]
            )
            bu = SeparatedReplayBuffer(
                self.all_args,
                self.envs.observation_space[agent_id],
                share_observation_space,
                self.envs.action_space[agent_id],
            )
            self.buffer.append(bu)
            self.trainer.append(tr)

    def run(self):
        raise NotImplementedError

    def warmup(self):
        raise NotImplementedError

    def collect(self, step):
        raise NotImplementedError

    def insert(self, data):
        raise NotImplementedError

    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(
                self.buffer[agent_id].share_obs[-1],
                self.buffer[agent_id].rnn_states_critic[-1],
                self.buffer[agent_id].masks[-1],
            )
            next_value = _t2n(next_value)
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)

    def train(self):
        train_infos = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_training()
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])
            train_infos.append(train_info)
            self.buffer[agent_id].after_update()

        return train_infos

    def save(self):
        for agent_id in range(self.num_agents):
            policy_actor = self.trainer[agent_id].policy.actor
            torch.save(
                policy_actor.state_dict(),
                str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt",
            )
            policy_critic = self.trainer[agent_id].policy.critic
            torch.save(
                policy_critic.state_dict(),
                str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt",
            )

    def restore(self):
        for agent_id in range(self.num_agents):
            policy_actor_state_dict = torch.load(str(self.model_dir) + "/actor_agent" + str(agent_id) + ".pt")
            self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
            policy_critic_state_dict = torch.load(
                str(self.model_dir) + "/critic_agent" + str(agent_id) + ".pt"
            )
            self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps):
        now_ts = time.time()
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                # 写 TensorBoard
                self.writter.add_scalars(agent_k, {agent_k: v}, total_num_steps)
                # 追加到 CSV（episode 暂无，写空或估算都可）
                try:
                    with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
                        csv.writer(f).writerow([now_ts, total_num_steps, "", agent_id, k, float(v)])
                except Exception:
                    pass

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                # if self.use_wandb:
                #     wandb.log({k: np.mean(v)}, step=total_num_steps)
                # else:
                self.writter.add_scalars(k, {k: np.mean(v)}, total_num_steps)

    def restore(self):
        for agent_id in range(self.num_agents):
            actor = torch.load(os.path.join(self.model_dir, f"actor_agent{agent_id}.pt"),
                               map_location=self.device)
            critic = torch.load(os.path.join(self.model_dir, f"critic_agent{agent_id}.pt"),
                                map_location=self.device)
            self.policy[agent_id].actor.load_state_dict(actor)
            self.policy[agent_id].critic.load_state_dict(critic)

    def log_train(self, train_infos, total_num_steps):
        now = time.time()
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                tag = f"agent{agent_id}/{k}"
                self.writter.add_scalar(tag, v, total_num_steps)
                # 追加到 CSV
                with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([f"{now:.3f}", total_num_steps, "", agent_id, k, v])

    def log_env(self, env_infos, total_num_steps):
        now = time.time()
        for k, v in env_infos.items():
            if len(v) > 0:
                self.writter.add_scalar(k, np.mean(v), total_num_steps)
                with open(self._csv_path, "a", newline="", encoding="utf-8") as f:
                    csv.writer(f).writerow([f"{now:.3f}", total_num_steps, "", -1, k, float(np.mean(v))])

    def should_validate(self, total_num_steps, final=False):
        if not self.use_eval:
            return False
        if final or total_num_steps >= self.next_validation_step:
            while self.next_validation_step <= total_num_steps:
                self.next_validation_step += (
                    self.validation_interval_steps
                )
            return True
        return False

    def log_validation(self, metrics, total_num_steps):
        row, report = self.validation_recorder.append(
            total_num_steps, metrics
        )
        for key in (
            "system_se_bps_hz",
            "sum_rate_mbps",
            "p05_rate_mbps",
            "qos_violation_rate",
            "overload_ratio",
        ):
            self.writter.add_scalar(
                f"validation/{key}",
                float(metrics.get(key, 0.0)),
                total_num_steps,
            )
        print(
            "VALIDATE-ORIG %d | steps=%d | episodes=%d | "
            "rate=%.2f Mbps | P05=%.2f Mbps | QoS=%.4f | "
            "plateau=%s | slope/100k=%.4f%%"
            % (
                row["validation_index"],
                total_num_steps,
                row["episodes"],
                row["sum_rate_mbps"],
                row["p05_rate_mbps"],
                row["qos_violation_rate"],
                report.get("plateau_detected", False),
                100.0
                * float(report.get("slope_relative_per_100k", 0.0)),
            ),
            flush=True,
        )
        return row, report
