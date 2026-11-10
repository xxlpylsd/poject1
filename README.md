# IICMS Mobility

本项目实现簇化 Cell-Free 网络中的干扰协调模式选择与 AP 簇选择。
当前主方法为 Controlled TF：共享 Transformer Actor、集中式 MLP Critic，
以及模式条件化的离散 AP 子集动作头。

## 项目结构

### 根目录

| 文件 | 作用 |
|---|---|
| `train_main.py` | 训练入口：构造环境、策略、Runner 和结果目录 |
| `config.py` | 全部命令行参数及其默认值 |
| `AGENTS.md` | 仓库逻辑、关键不变量、测试与 Pull Request 规则 |
| `run_iicms_comparison.sh` / `.ps1` | 一键训练、validation、checkpoint 选择、test、绘图和打包 |
| `evaluate_iicms.py` | 评价学习策略、固定动作和 RSSI 策略，并逐 episode 输出进度 |
| `evaluate_existing_runs.sh` | 对已有训练目录重新执行评价 |
| `select_best_checkpoint.py` | 用固定 validation seeds 选择最佳 checkpoint |
| `tune_rssi_thresholds.py` | 在 validation seeds 上校准 SC/DC-RSSI 门限 |
| `run_baseline_suite.sh` | 运行固定模式、动态聚类和 RSSI 基线套件 |
| `summarize_baseline_results.py` | 汇总基线 JSON 为 CSV/JSON |
| `evaluate_ap_count_sweep.py` | 不同 AP 数量下的 zero-shot 扩展评价，实时转发子评价进度 |
| `evaluate_backhaul_capacity_sweep.py` | 不同回程容量下的扩展评价 |
| `plot_training_curves.py` | 绘制训练或 validation 多 seed 曲线 |
| `plot_reward_convergence.py` | 由 `reward_proposed.npy` 绘制论文式 reward 收敛图 |
| `plot_strategy_cdf.py` | 绘制策略级 per-UE rate CDF |
| `plot_seed_se_cdf.py` | 绘制多 seed SE CDF |
| `plot_ap_count_results.py` | 绘制 AP 数量扩展结果 |
| `package_results.py` | 打包精简或完整实验结果 |
| `package_project.py` | 更新源码 SHA256 并打包源码和结果 |
| `channel.npz` | 默认信道/场景数据 |
| `requirements.txt` | Python 依赖 |
| `results/` | 训练、checkpoint、validation、test 和基线输出 |
| `analysis_outputs/` | 聚合分析和扩展实验输出 |
| `结果图/` | 论文和汇报图片 |

### `algorithms/`

| 路径 | 作用 |
|---|---|
| `algorithm/r_mappo.py` | PPO/GAE 更新、价值损失和联合动作更新 |
| `algorithm/rMAPPOPolicy.py` | Actor、Critic 和优化器封装 |
| `algorithm/r_actor_critic.py` | Actor/Critic 构造及动作头分派 |
| `utils/act.py` | gated、flat、factorized、subset 等动作头 |
| `utils/mlp.py` | MLP backbone |
| `utils/transformer.py` | Transformer backbone |
| `utils/rnn.py` | RNN 组件 |
| `utils/popart.py` | PopArt 奖励归一化 |
| `utils/distributions.py` | 分类和连续动作分布 |

### `envs/`

| 路径 | 作用 |
|---|---|
| `env_iicms_mobile.py` | 移动 IICMS 主环境、SINR/速率/回程/队列/QoS 计算 |
| `env_core.py` | 基础场景和信道模型 |
| `env_wrappers.py` | Dummy/Subprocess 向量环境和指标汇总 |
| `env_discrete.py` | 离散动作环境封装 |
| `env_continuous.py` | 连续动作环境封装 |

### `runner/`

| 路径 | 作用 |
|---|---|
| `shared/env_runner.py` | 共享策略 MAPPO 训练、validation、日志和保存 |
| `shared/base_runner.py` | 共享策略公共运行逻辑 |
| `separated/env_runner.py` | 每个 UE 独立策略的训练流程 |
| `separated/base_runner.py` | 独立策略公共运行逻辑 |

### `utils/`

| 文件 | 作用 |
|---|---|
| `policy_evaluation.py` | 共用 episode 评价、validation 曲线和平台期统计 |
| `shared_buffer.py` | 共享策略 PPO buffer |
| `separated_buffer.py` | 独立策略 PPO buffer |
| `valuenorm.py` | running mean/std 价值归一化 |
| `channel_api.py` | 信道场景生成、保存和加载 |
| `runtime_env.py` | 线程和运行环境设置 |
| `util.py` | 通用损失、梯度范数和初始化工具 |

### `tests/`

包含环境、动作头、MAPPO、评测协议、checkpoint、RSSI、绘图和 packaging 测试。

## 当前建模设置

### 网络与移动性

| 参数 | 设置 |
|---|---|
| 场景 | `IICMSMobility / PureMobility` |
| 区域 | `1 km × 1 km` |
| AP 数量 | 30 |
| UE 数量 | 30 |
| 每 AP 天线数 | 4 |
| AP 随机部署最小间距 | 50 m |
| 载频 | 1.9 GHz |
| 系统带宽 | 20 MHz |
| 每 AP 发射功率 | 0.2 W |
| 移动模型 | Random Waypoint |
| UE 速度 | 15 m/s |
| 决策时隙 | 0.5 s |
| 路径损耗 | legacy three-slope |
| Shadowing | 启用 |
| CSI aging | 关闭 |

### 干扰协调动作

| 模式 | AP 服务簇 | 候选簇大小 |
|---|---|---|
| JT | 多个 AP 联合传输 | 2、3、4、5 |
| CB | 一个主 AP 服务，其他 AP 协同抑制干扰 | 2、3、4、5 |
| NC | 不协作 | 1 |

AP 簇从每个 UE 的最强候选 AP 中选取。当前完整动作空间包含 53 个合法联合动作，
包括 JT/CB 的多种 AP 子集和 NC。

### 观测与网络输入

| 参数 | 设置 |
|---|---|
| UE token 维度 | 7 |
| 每帧 token | 1 个 UE + 5 个候选 AP |
| 历史帧数 | 4 |
| Actor 输入 | `24 × 7 = 168` 维 |
| Critic 输入 | 30 个 UE 的全局状态，5040 维 |
| AP 候选方式 | `subset` |
| 聚类策略 | 动态排序 |
| 动作选择 | 模式 + 簇大小 + AP 子集 |

### 回程与前传

| 参数 | 设置 |
|---|---|
| 每 AP 回程容量 | 550–650 Mbps 随机 |
| JT 回程模型 | bottleneck |
| 前传队列 | 有限队列 |
| 队列容量 | 0.5 s × 服务速率 |
| 队列溢出 | 计入有效速率、丢包和 QoS 指标 |

### 奖励与 QoS

当前主实验使用纯系统总速率奖励：

```text
r(t) = sum_k R_k(t)
```

| 参数 | 设置 |
|---|---|
| `reward_mode` | `team_rate` |
| 单位 | Mbps |
| QoS 惩罚权重 | 0 |
| UE 最低速率要求 | 40 Mbps |
| QoS 指标 | 违约率、速率缺口、P05 |
| 折扣因子 | 0.99 |
| GAE λ | 0.95 |

纯速率实验不对 QoS 施加训练惩罚；QoS 和 P05 只作为评价指标。

### 当前 MAPPO 策略设置

当前主对照是 Native shared MLP 与 Controlled TF。旧版 Native TF 不属于当前主方法。

| 参数 | Native shared MLP | Controlled TF |
|---|---:|---:|
| 策略共享 | 是 | 是 |
| Actor | MLP | Transformer |
| Critic | MLP | MLP |
| Actor 动作头 | `mlp_factorized` | `subset_conditional` |
| Actor 学习率 | `2e-4` | `1e-4` |
| Critic 学习率 | `1e-4` | `1e-4` |
| 熵系数 | `0.005` | `0.005` |
| Actor hidden size | 192 | 128 |
| Critic hidden size | 128 | 128 |
| Transformer d_model | — | 64 |
| Attention heads | — | 4 |
| Transformer layers | — | 2 |
| FFN multiplier | — | 4 |
| Dropout | — | 0 |
| Pooling | — | mean |
| Positional encoding | — | 启用 |
| Temporal context | `history_mean` | `history_mean` |

Controlled TF 使用 4 帧历史。每帧包含 1 个 UE token 和 5 个候选 AP token，
每个 token 为 7 维，因此 Actor 输入为 `24 × 7 = 168` 维。集中式 MLP Critic
使用 30 个 UE 的联合观测，共 `30 × 168 = 5040` 维。

动作采用 `subset_conditional`：

```text
协作模式：JT / CB / NC
簇大小：2 / 3 / 4 / 5
AP 子集：从每个 UE 当前最强的 5 个候选 AP 中选择
```

对应动作空间为：

```text
JT 的 26 个 AP 子集 + CB 的 26 个 AP 子集 + NC = 53 个合法联合动作
```

当前主实验固定使用 5 个候选 AP，不包含候选 AP 数量扩展或 anchored subset
实验。

共同训练设置：

| 参数 | 设置 |
|---|---:|
| PPO epochs | 5 |
| Mini-batch 数 | 1 |
| Clip 参数 | 0.2 |
| 并行训练环境 | 16 |
| Episode/rollout 长度 | 100 |
| 最终训练预算 | 600k 环境步 |
| 训练 seeds | 1–6 |
| 折扣因子 γ | 0.99 |
| GAE λ | 0.95 |
| Max gradient norm | 10 |

### Validation、checkpoint 与 test

| 阶段 | 设置 |
|---|---|
| 训练中 validation | 每 25k 环境步 |
| Validation episodes | 20 |
| Validation seeds | 91000–91019 |
| Validation horizon | 100 |
| Plateau window | 5 |
| Plateau tolerance | 0.5%/100k steps |
| Checkpoint 指标 | `system_se` |
| 每候选 checkpoint episodes | 20 |
| 最终 test seeds | 93000–93019 |
| 最终 test episodes | 20 |
| 执行方式 | 确定性 argmax 为主，随机执行单独报告 |

### RSSI 基线

RSSI 基线根据：

```text
ratio = best_link / sum(other_links)
```

选择 JT、CB 或 NC。门限只在 validation seeds 92000–92019 上校准，
测试 seeds 不参与门限选择。

当前高门限搜索范围：

```text
0.5, 1, 2, 4, 8, 16, 32, 64, 128, 256
```

## 快速运行

安装依赖：

```bash
pip install -r requirements.txt
```

运行测试：

```bash
python -m unittest discover -s tests -v
```

训练入口：

```bash
bash run_iicms_comparison.sh
```

参数设置：

```bash
python train_main.py --help
python evaluate_iicms.py --help
```

## 共同开发

本仓库欢迎修改和提交：

1. Fork `xxlpylsd/poject1`；
2. 在自己的分支中修改代码；
3. 运行 `python -m unittest discover -s tests -v`；
4. 提交 Pull Request；
5. 维护者审查后合并到 `main`。

仓库内部实现、数据流和修改约束见 [AGENTS.md](AGENTS.md)。

公开仓库允许其他人查看、克隆、Fork 和提交 Pull Request，但不会自动授予
直接写原仓库的权限。需要直接 push 的协作者，应由仓库所有者在
`Settings → Collaborators` 中添加并授予写权限。

当前仓库尚未声明 `LICENSE`。如果需要明确允许他人再发布、商用或闭源衍生，
应另行选择 MIT、Apache-2.0 或 GPL 等许可证。
