import argparse


def get_config():
    """
    The configuration parser for common hyperparameters of all environment.
    Please reach each `scripts/train/<env>_runner.py` file to find private hyperparameters
    only used in <env>.

    Prepare parameters:
        --algorithm_name <algorithm_name>
            specifiy the algorithm, including `["rmappo", "mappo", "rmappg", "mappg", "trpo"]`
        --experiment_name <str>
            an identifier to distinguish different experiment.
        --seed <int>
            set seed for numpy and torch
        --cuda
            by default True, will use GPU to train; or else will use CPU;
        --cuda_deterministic
            by default, make sure random seed effective. if set, bypass such function.
        --n_training_threads <int>
            number of training threads working in parallel. by default 1
        --n_rollout_threads <int>
            number of parallel envs for training rollout. by default 32
        --n_eval_rollout_threads <int>
            number of parallel envs for evaluating rollout. by default 1
        --n_render_rollout_threads <int>
            number of parallel envs for rendering, could only be set as 1 for some environments.
        --num_env_steps <int>
            number of env steps to train (default: 10e6)
        --user_name <str>
            [for wandb usage], to specify user's name for simply collecting training data.
        --use_wandb
            [for wandb usage], by default True, will log date to wandb server. or else will use tensorboard to log data.

    Env parameters:
        --env_name <str>
            specify the name of environment
        --use_obs_instead_of_state
            [only for some env] by default False, will use global state; or else will use concatenated local obs.

    Replay Buffer parameters:
        --episode_length <int>
            the max length of episode in the buffer.

    Network parameters:
        --share_policy
            by default True, all agents will share the same network; set to make training agents use different policies.
        --use_centralized_V
            by default True, use centralized training mode; or else will decentralized training mode.
        --stacked_frames <int>
            Number of input frames which should be stack together.
        --hidden_size <int>
            Dimension of hidden layers for actor/critic networks
        --layer_N <int>
            Number of layers for actor/critic networks
        --use_ReLU
            by default True, will use ReLU. or else will use Tanh.
        --use_popart
            by default True, use PopArt to normalize rewards.
        --use_valuenorm
            by default True, use running mean and std to normalize rewards.
        --use_feature_normalization
            by default True, apply layernorm to normalize inputs.
        --use_orthogonal
            by default True, use Orthogonal initialization for weights and 0 initialization for biases. or else, will use xavier uniform inilialization.
        --gain
            by default 0.01, use the gain # of last action layer
        --use_naive_recurrent_policy
            by default False, use the whole trajectory to calculate hidden states.
        --use_recurrent_policy
            by default, use Recurrent Policy. If set, do not use.
        --recurrent_N <int>
            The number of recurrent layers ( default 1).
        --data_chunk_length <int>
            Time length of chunks used to train a recurrent_policy, default 10.

    Optimizer parameters:
        --lr <float>
            learning rate parameter,  (default: 5e-4, fixed).
        --critic_lr <float>
            learning rate of critic  (default: 5e-4, fixed)
        --opti_eps <float>
            RMSprop optimizer epsilon (default: 1e-5)
        --weight_decay <float>
            coefficience of weight decay (default: 0)

    PPO parameters:
        --ppo_epoch <int>
            number of ppo epochs (default: 15)
        --use_clipped_value_loss
            by default, clip loss value. If set, do not clip loss value.
        --clip_param <float>
            ppo clip parameter (default: 0.2)
        --num_mini_batch <int>
            number of batches for ppo (default: 1)
        --entropy_coef <float>
            entropy term coefficient (default: 0.01)
        --use_max_grad_norm
            by default, use max norm of gradients. If set, do not use.
        --max_grad_norm <float>
            max norm of gradients (default: 0.5)
        --use_gae
            by default, use generalized advantage estimation. If set, do not use gae.
        --gamma <float>
            discount factor for rewards (default: 0.99)
        --gae_lambda <float>
            gae lambda parameter (default: 0.95)
        --use_proper_time_limits
            by default, the return value does consider limits of time. If set, compute returns with considering time limits factor.
        --use_huber_loss
            by default, use huber loss. If set, do not use huber loss.
        --use_value_active_masks
            by default True, whether to mask useless data in value loss.
        --huber_delta <float>
            coefficient of huber loss.

    PPG parameters:
        --aux_epoch <int>
            number of auxiliary epochs. (default: 4)
        --clone_coef <float>
            clone term coefficient (default: 0.01)

    Run parameters：
        --use_linear_lr_decay
            by default, do not apply linear decay to learning rate. If set, use a linear schedule on the learning rate

    Save & Log parameters:
        --save_interval <int>
            time duration between contiunous twice models saving.
        --log_interval <int>
            time duration between contiunous twice log printing.

    Eval parameters:
        --use_eval
            by default, do not start evaluation. If set`, start evaluation alongside with training.
        --eval_interval <int>
            time duration between contiunous twice evaluation progress.
        --eval_episodes <int>
            number of episodes of a single evaluation.

    Render parameters:
        --save_gifs
            by default, do not save render video. If set, save video.
        --use_render
            by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.
        --render_episodes <int>
            the number of episodes to render a given env
        --ifi <float>
            the play interval of each rendered image in saved video.

    Pretrained parameters:
        --model_dir <str>
            by default None. set the path to pretrained model.
    """



    parser = argparse.ArgumentParser(
        description="onpolicy", formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # === 新增：环境形态与场景规模 ===
    parser.add_argument("--action_space", type=str, default="discrete",
                        choices=["discrete", "continuous"],
                        help="动作空间：discrete 或 continuous")
    parser.add_argument("--num_aps", type=int, default=30, help="AP 数量")
    parser.add_argument("--antennas_per_ap", type=int, default=4, help="每个 AP 的天线数（论文取 4）")
    parser.add_argument("--map_size", type=float, default=1000.0, help="地图边长(米)")
    parser.add_argument(
        "--backhaul_lo",
        type=float,
        default=4e8,
        help="AP backhaul capacity lower bound in bit/s",
    )
    parser.add_argument(
        "--backhaul_hi",
        type=float,
        default=5e8,
        help="AP backhaul capacity upper bound in bit/s",
    )
    parser.add_argument(
        "--reward_mode",
        type=str,
        default="mean_se",
        choices=(
            "mean_se",
            "mean_rate",
            "team_rate",
            "paper",
            "team_sum",
            "per_ue",
        ),
        help=(
            "mean_se is the per-step average UE spectral efficiency used to "
            "optimize a time-averaged SE objective; mean_rate and team_rate "
            "use Mbps; paper is an ablation"
        ),
    )
    parser.add_argument(
        "--objective_horizon",
        type=int,
        default=100,
        help="decision steps used for the time-averaged SE objective",
    )
    parser.add_argument("--reward_phi1", type=float, default=0.5)
    parser.add_argument("--reward_phi2", type=float, default=1.0)
    parser.add_argument(
        "--rate_req_lo_mbps",
        type=float,
        default=40.0,
        help="minimum instantaneous UE rate in Mbps",
    )
    parser.add_argument(
        "--rate_req_hi_mbps",
        type=float,
        default=40.0,
        help="upper end of the sampled minimum-rate requirement",
    )
    parser.add_argument(
        "--min_ue_se_bps_hz",
        type=float,
        default=0.0,
        help=(
            "soft safety reporting floor for each UE's SE; it does not "
            "change the SE reward"
        ),
    )
    parser.add_argument(
        "--qos_penalty_weight",
        type=float,
        default=20.0,
        help="weight of the minimum-rate constraint penalty",
    )
    parser.add_argument(
        "--backhaul_jt_model",
        type=str,
        default="bottleneck",
        choices=("product", "bottleneck"),
        help=(
            "product keeps legacy multiplicative AP scaling; bottleneck uses "
            "the worst serving-AP scale for a JT UE"
        ),
    )
    parser.add_argument(
        "--cluster_sizes",
        type=str,
        default="2,3,4,5",
        help="CB/JT candidate cluster sizes, e.g. 3,4,5",
    )
    parser.add_argument(
        "--cluster_action_mode",
        type=str,
        default="topk",
        choices=("topk", "subset"),
        help=(
            "topk keeps the strongest top-L APs as a strong heuristic "
            "baseline; subset lets the policy select the AP membership "
            "pattern among the strongest max(cluster_sizes) candidates"
        ),
    )
    parser.add_argument(
        "--cluster_policy",
        type=str,
        default="dynamic",
        choices=("static", "dynamic"),
        help=(
            "dynamic refreshes the candidate AP ranking at every slot; "
            "static freezes the ranking and physical AP set at episode reset"
        ),
    )
    parser.add_argument(
        "--ap_min_distance",
        type=float,
        default=50.0,
        help="minimum pairwise AP spacing in metres",
    )
    parser.add_argument("--channel_file", type=str, default=None,
                        help="固定信道文件路径（.npz/.json），若提供则加载该场景")

    # === 纯移动性扩展 ===
    parser.add_argument("--mobility", type=str, default="rwp",
                        choices=["static", "rwp", "random_walk"],
                        help="UE 移动模型")
    parser.add_argument("--ue_speed", type=float, default=10.0,
                        help="UE 平均速度 (m/s)")
    parser.add_argument("--slot_dt", type=float, default=0.5,
                        help="每个决策时隙的时长 (s)")
    parser.add_argument("--reset_ues", type=int, default=1, choices=[0, 1],
                        help="每个 episode 是否重新采样 UE 位置")
    parser.add_argument("--pathloss_model", type=str, default="legacy",
                        choices=["legacy", "classic"],
                        help="legacy=原始代码（用于逐位复现）；classic=连续单调标准三段模型")
    parser.add_argument("--csi_aging", type=int, default=0, choices=[0, 1],
                        help="1=启用 Jakes 信道老化；0=关闭")
    parser.add_argument("--carrier_ghz", type=float, default=1.9,
                        help="载频 (GHz)")
    parser.add_argument("--csi_block_slots", type=int, default=200,
                        help="一个通信块的 slot 数")
    parser.add_argument("--csi_pilot_slots", type=int, default=10,
                        help="每个通信块的导频 slot 数")
    parser.add_argument("--csi_slot_ms", type=float, default=1.0,
                        help="每个 slot 的时长 (ms)")
    parser.add_argument(
        "--obs_include_xy",
        type=int,
        default=0,
        choices=[0, 1],
        help=(
            "append normalized absolute UE and AP coordinates to their "
            "respective tokens"
        ),
    )
    parser.add_argument(
        "--obs_frames",
        type=int,
        default=1,
        help="number of recent observation frames exposed to the actor",
    )
    parser.add_argument(
        "--fronthaul_queue",
        type=int,
        default=0,
        choices=(0, 1),
        help=(
            "enable a finite per-AP fluid queue on the AP-CPU fronthaul "
            "link instead of instantaneous clipping only"
        ),
    )
    parser.add_argument(
        "--fronthaul_buffer_s",
        type=float,
        default=1.0,
        help=(
            "finite queue capacity expressed as seconds at the AP "
            "fronthaul service rate"
        ),
    )

    # prepare parameters
    parser.add_argument("--algorithm_name", type=str, default="mappo", choices=["rmappo", "mappo"])

    parser.add_argument(
        "--experiment_name",
        type=str,
        default="check",
        help="an identifier to distinguish different experiment.",
    )
    parser.add_argument("--seed", type=int, default=1, help="Random seed for numpy/torch")
    parser.add_argument(
        "--cuda",
        action="store_false",
        default=True,
        help="by default True, will use GPU to train; or else will use CPU;",
    )
    parser.add_argument(
        "--cuda_deterministic",
        action="store_false",
        default=True,
        help="by default, make sure random seed effective. if set, bypass such function.",
    )
    parser.add_argument(
        "--n_training_threads",
        type=int,
        default=2,
        help="Number of torch threads for training",
    )
    parser.add_argument(
        "--n_rollout_threads",
        type=int,
        default=5,
        help="Number of parallel envs for training rollouts",
    )
    parser.add_argument(
        "--vec_env",
        type=str,
        default="auto",
        choices=("auto", "dummy", "subproc"),
        help="Use subprocess-parallel rollout envs on Linux; dummy is serial",
    )
    parser.add_argument(
        "--n_eval_rollout_threads",
        type=int,
        default=2,
        help="Number of parallel envs for evaluating rollouts",
    )
    parser.add_argument(
        "--n_render_rollout_threads",
        type=int,
        default=1,
        help="Number of parallel envs for rendering rollouts",
    )
    parser.add_argument(
        "--num_env_steps",
        type=int,
        default=1e6,
        help="Number of environment steps to train (default: 10e6)",
    )
    parser.add_argument(
        "--user_name",
        type=str,
        default="marl",
        help="[for wandb usage], to specify user's name for simply collecting training data.",
    )

    # env parameters
    parser.add_argument("--env_name", type=str, default="MyEnv", help="specify the name of environment")
    parser.add_argument(
        "--use_obs_instead_of_state",
        action="store_true",
        default=False,
        help="Whether to use global state or concatenated obs",
    )

    # replay buffer parameters
    parser.add_argument("--episode_length", type=int, default=100, help="Max length for any episode")

    # network parameters
    parser.add_argument(
        "--share_policy",
        action="store_false",
        default=True,
        help="Whether agent share the same policy",
    )
    parser.add_argument(
        "--separate_policy",
        dest="share_policy",
        action="store_false",
        help=(
            "Create one independent Actor/Critic pair per UE; this is the "
            "original-paper MAPPO setting"
        ),
    )
    parser.add_argument(
        "--original_mappo",
        action="store_true",
        default=False,
        help=(
            "Use the original-paper-style comparison baseline: independent "
            "per-UE MLP actors, concatenated-state MLP critics, and "
            "factorized mode/cluster-size actions"
        ),
    )
    parser.add_argument(
        "--use_centralized_V",
        action="store_false",
        default=True,
        help="Whether to use centralized V function",
    )
    parser.add_argument(
        "--no_critic_dedup",
        dest="critic_dedup",
        action="store_false",
        default=True,
        help=(
            "Disable sharing the identical centralized observation across "
            "agents during critic evaluation"
        ),
    )
    parser.add_argument(
        "--stacked_frames",
        type=int,
        default=1,
        help="Dimension of hidden layers for actor/critic networks",
    )
    parser.add_argument(
        "--use_stacked_frames",
        action="store_true",
        default=False,
        help="Whether to use stacked_frames",
    )
    parser.add_argument(
        "--hidden_size",
        type=int,
        default=64,
        help="Dimension of hidden layers for actor/critic networks",
    )
    parser.add_argument("--actor_backbone", type=str, default="mlp",
                        choices=["mlp", "transformer"])
    parser.add_argument("--critic_backbone", type=str, default="mlp",
                        choices=["mlp", "transformer"])
    parser.add_argument(
        "--disable_actor_token_norm",
        dest="actor_token_norm",
        action="store_false",
        default=True,
        help=(
            "normalize MLP actor observations per structured token instead "
            "of over the flattened observation"
        ),
    )
    parser.add_argument(
        "--joint_action_head",
        type=str,
        default="gated",
        choices=(
            "gated",
            "flat",
            "subset",
            "subset_conditional",
            "relational_subset",
            "mlp_factorized",
            "paper_factorized",
        ),
        help=(
            "gated: two mode/size branches with token gating and one joint "
            "distribution; flat: ordinary categorical over joint actions; "
            "subset: score candidate AP tokens and enumerate valid L-subsets; "
            "subset_conditional additionally conditions cluster choice on mode; "
            "relational_subset adds Transformer-native attention pooling and "
            "cross-AP relation scoring; mlp_factorized uses an MLP encoder "
            "with factorized mode, size, and AP-member heads; "
            "paper_factorized reproduces the original paper's independent "
            "mode and cluster-size categorical branches with top-L execution"
        ),
    )
    parser.add_argument("--joint_num_modes", type=int, default=3)
    parser.add_argument("--joint_num_sizes", type=int, default=3)
    parser.add_argument(
        "--joint_gate_hidden",
        type=int,
        default=0,
        help="hidden width of the soft gate; 0 uses actor hidden size",
    )
    parser.add_argument("--tf_d_model", type=int, default=64)
    parser.add_argument(
        "--tf_relation_dim",
        type=int,
        default=32,
        help="projection width of the Transformer-native relational action head",
    )
    parser.add_argument("--tf_n_head", type=int, default=4)
    parser.add_argument("--tf_n_layers", type=int, default=2)
    parser.add_argument("--tf_ff_mult", type=int, default=4)
    parser.add_argument("--tf_dropout", type=float, default=0.0)
    parser.add_argument("--tf_pool", type=str, default="mean",
                        choices=["mean", "max", "cls"])
    parser.add_argument(
        "--actor_temporal_context",
        type=str,
        default="history_mean",
        choices=(
            "history_mean",
            "current_frame",
            "current_frame_balanced",
        ),
        help=(
            "context used by mode and cluster-size heads: mean of all "
            "history tokens, or mean of the current frame's contextualized "
            "UE/AP tokens; balanced gives UE and AP contexts equal weight"
        ),
    )
    parser.add_argument("--tf_use_cls", action="store_true", default=False)
    parser.add_argument("--tf_use_pe", action="store_true", default=True)
    parser.add_argument("--tf_max_tokens", type=int, default=256)
    parser.add_argument("--tf_grad_checkpoint", type=int, default=1,
                        choices=[0, 1])
    parser.add_argument("--critic_hidden_size", type=int, default=0,
                        help="Critic hidden width; 0 uses hidden_size")
    parser.add_argument(
        "--layer_N",
        type=int,
        default=1,
        help="Number of layers for actor/critic networks",
    )
    parser.add_argument("--use_ReLU", action="store_false", default=True, help="Whether to use ReLU")
    parser.add_argument(
        "--use_popart",
        action="store_true",
        default=False,
        help="by default False, use PopArt to normalize rewards.",
    )
    parser.add_argument(
        "--use_valuenorm",
        action="store_false",
        default=True,
        help="by default True, use running mean and std to normalize rewards.",
    )
    parser.add_argument(
        "--use_feature_normalization",
        action="store_false",
        default=True,
        help="Whether to apply layernorm to the inputs",
    )
    parser.add_argument(
        "--use_orthogonal",
        action="store_false",
        default=True,
        help="Whether to use Orthogonal initialization for weights and 0 initialization for biases",
    )
    parser.add_argument("--gain", type=float, default=0.01, help="The gain # of last action layer")

    # recurrent parameters
    parser.add_argument(
        "--use_naive_recurrent_policy",
        action="store_true",
        default=False,
        help="Whether to use a naive recurrent policy",
    )
    parser.add_argument(
        "--use_recurrent_policy",
        action="store_false",
        default=False,
        help="use a recurrent policy",
    )
    parser.add_argument("--recurrent_N", type=int, default=1, help="The number of recurrent layers.")
    parser.add_argument(
        "--data_chunk_length",
        type=int,
        default=10,
        help="Time length of chunks used to train a recurrent_policy",
    )

    # optimizer parameters
    parser.add_argument("--lr", type=float, default=5e-5, help="learning rate (default: 5e-4)")
    parser.add_argument(
        "--critic_lr",
        type=float,
        default=5e-5,
        help="critic learning rate (default: 5e-4)",
    )
    parser.add_argument(
        "--opti_eps",
        type=float,
        default=1e-5,
        help="RMSprop optimizer epsilon (default: 1e-5)",
    )
    parser.add_argument("--weight_decay", type=float, default=0)

    # ppo parameters
    parser.add_argument("--ppo_epoch", type=int, default=15, help="number of ppo epochs (default: 15)")
    parser.add_argument(
        "--use_clipped_value_loss",
        action="store_false",
        default=True,
        help="by default, clip loss value. If set, do not clip loss value.",
    )
    parser.add_argument(
        "--clip_param",
        type=float,
        default=0.2,
        help="ppo clip parameter (default: 0.2)",
    )
    parser.add_argument(
        "--num_mini_batch",
        type=int,
        default=1,
        help="number of batches for ppo (default: 1)",
    )
    parser.add_argument(
        "--entropy_coef",
        type=float,
        default=0.01,
        help="entropy term coefficient (default: 0.01)",
    )
    parser.add_argument(
        "--no_joint_action_ppo",
        dest="joint_action_ppo",
        action="store_false",
        default=True,
        help=(
            "Use independent per-head PPO clipping instead of the joint "
            "MultiDiscrete action probability"
        ),
    )
    parser.add_argument(
        "--value_loss_coef",
        type=float,
        default=1,
        help="value loss coefficient (default: 0.5)",
    )
    parser.add_argument(
        "--use_max_grad_norm",
        action="store_false",
        default=True,
        help="by default, use max norm of gradients. If set, do not use.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=10.0,
        help="max norm of gradients (default: 0.5)",
    )
    parser.add_argument(
        "--use_gae",
        action="store_false",
        default=True,
        help="use generalized advantage estimation",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help=(
            "discount factor; default 0.99 has effective memory about "
            "100 decisions, matching the default objective horizon"
        ),
    )
    parser.add_argument(
        "--gae_lambda",
        type=float,
        default=0.95,
        help="gae lambda parameter (default: 0.95)",
    )
    parser.add_argument(
        "--use_proper_time_limits",
        action="store_true",
        default=False,
        help="compute returns taking into account time limits",
    )
    parser.add_argument(
        "--use_huber_loss",
        action="store_false",
        default=True,
        help="by default, use huber loss. If set, do not use huber loss.",
    )
    parser.add_argument(
        "--use_value_active_masks",
        action="store_false",
        default=True,
        help="by default True, whether to mask useless data in value loss.",
    )
    parser.add_argument(
        "--use_policy_active_masks",
        action="store_false",
        default=True,
        help="by default True, whether to mask useless data in policy loss.",
    )
    parser.add_argument("--huber_delta", type=float, default=10.0, help=" coefficience of huber loss.")

    # run parameters
    parser.add_argument(
        "--use_linear_lr_decay",
        action="store_true",
        default=False,
        help="use a linear schedule on the learning rate",
    )
    # save parameters
    parser.add_argument(
        "--save_interval",
        type=int,
        default=1,
        help="time duration between contiunous twice models saving.",
    )
    parser.add_argument(
        "--save_checkpoint_snapshots",
        action="store_true",
        default=False,
        help=(
            "also retain a step-numbered actor/critic snapshot at every "
            "save interval so validation can select a non-final checkpoint"
        ),
    )

    # log parameters
    parser.add_argument(
        "--log_interval",
        type=int,
        default=5,
        help="time duration between contiunous twice log printing.",
    )

    # eval parameters
    parser.add_argument(
        "--use_eval",
        action="store_true",
        default=False,
        help="by default, do not start evaluation. If set`, start evaluation alongside with training.",
    )
    parser.add_argument(
        "--eval_interval",
        type=int,
        default=25,
        help="time duration between contiunous twice evaluation progress.",
    )
    parser.add_argument(
        "--eval_episodes",
        type=int,
        default=32,
        help="number of episodes of a single evaluation.",
    )

    # render parameters
    parser.add_argument(
        "--save_gifs",
        action="store_true",
        default=False,
        help="by default, do not save render video. If set, save video.",
    )
    parser.add_argument(
        "--use_render",
        action="store_true",
        default=False,
        help="by default, do not render the env during training. If set, start render. Note: something, the environment has internal render process which is not controlled by this hyperparam.",
    )
    parser.add_argument(
        "--render_episodes",
        type=int,
        default=5,
        help="the number of episodes to render a given env",
    )
    parser.add_argument(
        "--ifi",
        type=float,
        default=0.1,
        help="the play interval of each rendered image in saved video.",
    )

    # pretrained parameters
    parser.add_argument(
        "--model_dir",
        type=str,
        default=None,
        help="by default None. set the path to pretrained model.",
    )

    return parser
