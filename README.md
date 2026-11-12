## 文件说明

| 文件或目录 | 作用 |
|---|---|
| `train_main.py` | 训练入口，创建环境、策略、Runner 和运行目录 |
| `config.py` | 命令行参数及默认值 |
| `run_contention_600k.sh` | Linux/WSL 当前主实验：同一种子训练 MLP 与 TF，并完成验证、选模和测试 |
| `run_contention_600k.ps1` | Windows PowerShell 下的当前主实验入口 |
| `resume_contention_600k.sh` | 中断后复用已有 MLP，继续 checkpoint 选择或 TF 训练 |
| `run_iicms_comparison.sh` / `.ps1` | 可配置的完整训练与评价流程 |
| `evaluate_iicms.py` | 评价学习策略及固定动作、RSSI 基线 |
| `evaluate_existing_runs.sh` | 重新评价已有训练目录 |
| `select_best_checkpoint.py` | 使用固定验证种子选择 checkpoint |
| `tune_rssi_thresholds.py` | 在验证集上校准 RSSI 门限 |
| `run_baseline_suite.sh` | 运行完整启发式基线套件 |
| `summarize_baseline_results.py` | 汇总基线评价结果 |
| `evaluate_ap_count_sweep.py` | AP 数量 zero-shot 扩展评价 |
| `evaluate_backhaul_capacity_sweep.py` | 前传容量 zero-shot 扩展评价 |
| `plot_*.py` | 训练曲线、reward 收敛和 per-UE CDF 绘图 |
| `package_project.py` / `.sh` / `.ps1` | 生成源码清单与本地交付包 |
| `package_results.py` | 打包实验结果 |
| `channel.npz` | 默认信道与场景数据 |
| `requirements.txt` | Python 依赖 |
| `figures/` | 最终 AP、前传、CDF 和训练曲线展示图 |
| `AGENTS.md` | 数据流、建模不变量、测试和协作规则 |

## 核心代码

| 路径 | 作用 |
|---|---|
| `envs/env_iicms_mobile.py` | 移动性环境、观测、SINR、速率、回程、队列和 QoS |
| `envs/env_core.py` | 信道、部署与基础速率模型 |
| `envs/env_wrappers.py` | Dummy/Subprocess 向量环境 |
| `algorithms/algorithm/r_mappo.py` | GAE 与 PPO 更新 |
| `algorithms/algorithm/r_actor_critic.py` | Actor/Critic 构造 |
| `algorithms/utils/act.py` | 联合动作分布和动作头 |
| `algorithms/utils/mlp.py` | MLP backbone |
| `algorithms/utils/transformer.py` | Transformer backbone |
| `runner/shared/` | 共享策略 MAPPO 训练循环 |
| `runner/separated/` | 独立策略 MAPPO 训练循环 |
| `utils/policy_evaluation.py` | 验证、checkpoint 指标和平台期统计 |
| `utils/shared_buffer.py` | 共享策略经验缓存 |
| `utils/separated_buffer.py` | 独立策略经验缓存 |
| `tests/` | 环境、动作、训练协议、评价和绘图测试 |

## 当前建模设置

### 网络与移动性

| 参数 | 设置 |
|---|---|
| 场景 | `IICMSMobility / PureMobility` |
| 区域 | 1 km × 1 km |
| AP / UE | 30 / 30 |
| 每 AP 天线数 | 4 |
| AP 最小间距 | 50 m |
| 载频 / 带宽 | 1.9 GHz / 20 MHz |
| 每 AP 发射功率 | 0.2 W |
| 移动模型 | Random Waypoint |
| UE 速度 | 15 m/s |
| 决策时隙 | 0.5 s |
| Shadowing | 启用 |
| CSI aging | 主实验关闭 |

### 动作空间

| 模式 | 含义 | 候选簇大小 |
|---|---|---|
| JT | 多 AP 联合传输 | 2、3、4、5 |
| CB | 主 AP 服务，其他 AP 协同抑制干扰 | 2、3、4、5 |
| NC | 不协作 | 1 |

AP 从每个 UE 当前最强的 5 个候选中选择。完整动作空间为
`26 (JT) + 26 (CB) + 1 (NC) = 53` 个合法联合动作。

### 回程、前传与奖励

| 参数 | 设置 |
|---|---|
| 每 AP 回程容量 | 550–650 Mbps 随机 |
| JT 回程模型 | bottleneck |
| 前传队列 | 有限队列，容量 0.5 s × 服务速率 |
| 队列溢出 | 计入有效速率、丢包和 QoS 指标 |
| 奖励 | `team_rate`，系统总速率 |
| 单位 | Mbps |
| QoS 惩罚权重 | 0 |
| QoS 最低速率要求 | 40 Mbps |
| 折扣因子 / GAE λ | 0.99 / 0.95 |

主实验为纯速率训练，QoS 与 P05 仅作为评价指标。

### 策略

| 参数 | MLP | TF |
|---|---:|---:|
| 策略共享 | 是 | 是 |
| Actor | MLP | Transformer |
| Critic | MLP | MLP |
| Actor 学习率 | `2e-4` | `1e-4` |
| Critic 学习率 | `1e-4` | `1e-4` |
| 熵系数 | `0.005` | `0.005` |
| Actor hidden | 192 | 128 |
| Critic hidden | 128 | 128 |
| Transformer d_model | — | 64 |
| Attention heads / layers | — | 4 / 2 |
| FFN multiplier | — | 4 |
| Dropout | — | 0 |
| Pooling | — | mean |

共同训练设置：

| 参数 | 设置 |
|---|---:|
| PPO epochs | 5 |
| Mini-batch 数 | 1 |
| Clip | 0.2 |
| 并行训练环境 | Linux 16；Windows 默认顺序执行 |
| Episode / rollout 长度 | 100 |
| 训练预算 | 600k 环境步 |
| 训练 seeds | 1–6 |
| 梯度裁剪 | 10 |

## 验证与测试

| 阶段 | 设置 |
|---|---|
| 训练中 validation | 每 25k 环境步 |
| Validation episodes | 20 |
| Validation seeds | 91000–91019 |
| 高门限 validation seeds | 92000–92019 |
| Validation horizon | 100 |
| Plateau window | 5 |
| Plateau tolerance | 每次 100k 步相对增长低于 0.5% |
| Checkpoint 指标 | `system_se` |
| Checkpoint episodes | 20 |
| 最终 test seeds | 93000–93019 |
| 最终 test episodes | 20 |
| 主结果执行方式 | 确定性 argmax |
| 补充结果执行方式 | 随机策略采样 |

验证、门限调参和最终测试使用互不重叠的种子集合。训练日志和 validation
曲线用于判断收敛；最终性能不能与训练曲线直接按数值比较。

## 本地 VS Code 运行

本地运行不需要 AutoDL。Windows、Linux、macOS 和 WSL 均可运行；有
NVIDIA GPU 时会自动使用 CUDA，没有 GPU 时自动回退到 CPU。

### 1. 获取代码

```bash
git clone https://github.com/xxlpylsd/poject1.git
cd poject1
```

也可以在 VS Code 中执行 `Git: Clone`，然后打开仓库目录。

### 2. 创建 Python 环境

建议使用 Python 3.9–3.11。Windows PowerShell：

```powershell
py -3.11 -m venv .venv
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Linux、macOS 或 WSL：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

需要 GPU 时，先确认 PyTorch 能识别显卡：

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

### 3. 运行测试

在仓库根目录执行：

```bash
python -m unittest discover -s tests -v
```

全套测试不要求 GPU，正常情况下应全部通过。

### 4. Windows 短测

Windows 没有 `fork`，程序会自动把向量环境切换为串行 `dummy` 模式。
先运行 800 步 CPU 冒烟测试：

```powershell
.\run_contention_600k.ps1 -Seed 1 -Smoke
```

短测会依次完成一次 MLP 和 TF 的 rollout 与 PPO 更新。GPU 冒烟测试可去掉
CPU 回退：

```powershell
.\run_contention_600k.ps1 -Seed 1 -Smoke -Cpu:$false
```

### 5. Windows 正式训练

同一种子训练 MLP 和 TF：

```powershell
.\run_contention_600k.ps1 -Seed 1 -Steps 600000 -Envs 4
```

只训练其中一个模型：

```powershell
.\run_contention_600k.ps1 -Seed 1 -Steps 600000 -Algos mlp
.\run_contention_600k.ps1 -Seed 1 -Steps 600000 -Algos tf
```

强制使用 CPU：

```powershell
.\run_contention_600k.ps1 -Seed 1 -Steps 600000 -Cpu
```

如果使用其他 Python 环境，可通过 `-Python` 指定解释器：

```powershell
.\run_contention_600k.ps1 -Seed 1 -Python "C:\path\to\python.exe"
```

Windows 的 `dummy` 环境会顺序执行多个 rollout 环境，速度慢于 Linux。
需要与服务器接近的并行速度时，建议在 VS Code 中连接 WSL。

### 6. Linux、macOS 或 WSL 正式训练

当前主实验脚本只接收种子和训练步数：

```bash
bash run_contention_600k.sh 1 600000
```

补齐其他种子：

```bash
for seed in 2 3 4 5 6; do
  bash run_contention_600k.sh "$seed" 600000 || exit 1
done
```

该脚本在 Linux 下默认使用 `subproc` 并行环境。中断后如需复用已完成的 MLP：

```bash
bash resume_contention_600k.sh 1 600000
```

### 7. 单独测试已有模型

训练完成后，默认最佳 checkpoint 位于：

```text
results/IICMSMobility/PureMobility/mappo/<实验名>_s<种子>/run1/models/
```

确定性测试示例：

```bash
python evaluate_iicms.py \
  --run_dir results/IICMSMobility/PureMobility/mappo/rate_contention_neighbor3_600000_tf_actor_s1/run1 \
  --seed_start 93000 \
  --episodes 20 \
  --horizon 100 \
  --phase test \
  --checkpoint best \
  --out results/manual_eval_tf_s1_deterministic
```

随机执行测试需要在同一命令中加入 `--stochastic`。测试种子必须与
validation/checkpoint 选择种子保持独立。

## 参数与结果位置

查看完整参数：

```bash
python train_main.py --help
python evaluate_iicms.py --help
```

默认结果目录：

```text
results/IICMSMobility/PureMobility/mappo/<实验名>_s<种子>/run1/
```

其中包含训练曲线、validation、checkpoint 选择、确定性/随机测试和模型权重。
这些实验输出已被 `.gitignore` 排除，不会误提交到 GitHub。
