# IICMS Gated Joint-Action Mobility

本目录是与 `clean_mobility_v1` 完全隔离的新实验。核心研究变量是 UE 移动；
动作包含干扰协调模式、协作簇大小和协作 AP 子集，并加入实际部署所需的
AP 最小间距和 UE 最低速率约束。

当前主实验使用每个 AP 独立的有限 AP-CPU 前传队列。开启队列历史后，单帧
观测为 `6 token × 7 dim = 42`，4 帧 Actor 观测为 `168` 维。旧版 144 维
checkpoint 与当前观测维度不兼容，不能和新环境结果混用；MLP 与 Transformer
必须从同一新环境重新训练。

## 0. 当前保留范围

工程已清理历史烟雾测试、失败实验、中间权重、TensorBoard事件和旧结果包。
当前保留的实验为：

- `fronthaul_queue_mlp_factorized_rate_400k_mlp_s1..s6`：新MLP主实验；
- `fronthaul_queue_native_rate_400k_mlp_s1..s6`：旧MLP对照；
- `fronthaul_queue_native_rate_400k_tf_actor_s1..s6`：TF主实验；
- `analysis_outputs/ap_count_zero_shot`：AP数量零样本扩展；
- `analysis_outputs/backhaul_capacity_zero_shot`：回程容量扩展；
- `analysis_outputs/20261106_rate_cdf_6seed`：CDF展示结果。

历史筛查、旧实验和被移除的过程文件已隔离到工程目录之外，可恢复；
当前训练、RSSI基线标定、测评、绘图和一键打包流程不依赖隔离内容。

## 1. 环境边界

环境包含：

- 原始 IICMS 的 JT/CB/NC 物理计算；
- 可配置的 top-L 强基线或可学习协作 AP 子集；
- 每 AP 有限服务率、跨时隙积压和有限缓冲溢出；
- Random Waypoint 移动；
- 每个决策时隙根据 UE 新位置重算距离和大尺度衰落；
- AP 两两间距不小于 50 m；
- 每个 UE 的瞬时速率不低于 40 Mbps 的软约束；
- 可选的 Jakes 块内信道老化，仅用于 A/B，不属于主实验默认设置。

环境不包含：

- HO keep/accept 动作；
- HO 中断代价或 HO 历史观测；
- CPU 簇切换代价；
- 位置阴影场、AP token 随机打乱或域随机化。

原始代码中的 CB 有一个物理一致性问题：所有非主 AP 都使用
`cb_others` 干扰增益，导致 CB 的簇大小完全不起作用。现在仅对协作簇内的
非主 AP 使用 `cb_others`，协作簇外使用普通干扰增益 `base`。这是让动作
与论文中 `M_C(k)` 协作边界保持一致所必需的修正。

## 2. 动作空间

`cluster_action_mode` 提供两个严格隔离的动作空间。

### top-L 强基线

```text
cluster_action_mode = topk
MultiDiscrete([7])
```

```text
= {JT_L2, JT_L3, JT_L5,
   CB_L2, CB_L3, CB_L5,
   NC}
```

协作簇固定取当前信道最强的前 L 个 AP。该模式保留用于复现前期实验，也是
移动性策略必须超过的强启发式基线。

### 可学习 AP 子集（主实验）

```text
cluster_action_mode = subset
cluster_sizes = 2,3,4,5
candidate AP pool = strongest Kmax=5 APs
```

对每个 CB/JT UE，动作同时决定：

```text
模式: JT / CB
簇大小: L=2 / 3 / 4 / 5
AP 子集: 从候选 5 个 AP 中选择 L 个
```

物理有效动作数为：

```text
JT: C(5,2)+C(5,3)+C(5,4)+C(5,5)
  = 10+10+5+1 = 26
CB: C(5,2)+C(5,3)+C(5,4)+C(5,5)
  = 10+10+5+1 = 26
NC: 1
总计: 53
```

动作编号形如 `JT_L3_S04`；`S00` 始终表示最强 top-L 子集。所有子集都以
相对信道排名表示，不依赖 AP 的全局编号，因此策略能够随 UE 移动重新选择
协作 AP。NC 仍只有一个动作。

环境配置层面的默认值保持 `topk`，以免旧 checkpoint 被错误地按新动作空间
加载；项目的一键学习对比脚本默认使用 `subset`。显式切回强基线：

```bash
CLUSTER_ACTION_MODE=topk bash run_iicms_comparison.sh
```

### 网络结构

主实验默认使用 `joint_action_head=subset_conditional`。它不是把 53 个编号当成 53 个
互不相关的类别，而是显式利用子集结构：

```text
mode logits: 3
candidate AP scores: 5 x 2（JT/CB 各有逐 AP 分数）
size logits: 4 x 2（JT/CB 各有簇大小偏好）
cluster-choice logits for mode m:
    mean(member AP scores in m) + size logit in m
joint distribution:
    P(mode) * P(cluster choice | mode)
```

因此，学习“某个候选 AP 是否有用”会迁移到所有包含该 AP 的子集；不会要求网络
分别记忆完整的26个互不相干编号。确定性推理也先选模式、再选簇大小/AP子集，
避免NC因只占一个动作槽而在53类argmax中获得人为优势。
簇选择条件在模式上也是必需的：当前物理模型表现为 JT 偏好小簇、CB 偏好大簇，
旧 `subset` 头共享一套簇 logits，无法表达这种交互。

Transformer 使用编码后的 AP token 计算逐 AP 分数；MLP 使用相同的逐 token
结构，通过共享 token 投影计算 AP 分数。两种算法只改变观测编码骨干。

Transformer Actor 新增可配置的`actor_temporal_context`：

```text
history_mean            ：模式/簇大小使用全部4帧token的平均特征（原行为）
current_frame           ：当前帧UE与5个AP token等权平均
current_frame_balanced  ：当前UE token与当前AP均值各占50%
```

`current_frame`不会丢弃历史：当前帧token已经通过自注意力读取前3帧，因此仍包含
时序信息，但不再被24个历史token的均值稀释。AP子集评分始终使用当前帧token。
该选项默认保持`history_mean`以兼容旧checkpoint；实验脚本通过
`TF_ACTOR_TEMPORAL_CONTEXT=current_frame`或
`current_frame_balanced`启用新Actor变体。后者避免5个AP token把当前UE的
SINR/QoS状态稀释到`1/6`权重。

保留的门控动作头用于历史复现：

```text
mode logits: 3
cluster logits: 26
joint interaction logits: 3 x 26
one softmax over 53 valid actions
```

可用 `JOINT_ACTION_HEAD=subset` 复现模式无关的旧子集头，或用
`JOINT_ACTION_HEAD=gated` / `--joint_action_head flat` 做其他动作头消融。

主实验根据前期筛选结果固定：

```text
cluster_sizes = 2,3,4,5
```

NC 不使用协作簇，因此只保留唯一的 NC 动作。

动作空间和簇大小筛查已在主实验前完成。对应的历史 `screening/` 脚本和
结果已从精简工程移入隔离目录，当前主流程不再依赖它们。

本次 3 条轨迹 × 20 步的配对筛查结果：

| 策略 | 系统 SE | 相对 top-L | QoS 违约率 |
|---|---:|---:|---:|
| top-L JT L=2 | 176.83 | 0.00% | 0.06% |
| random subset | 106.36 | -39.85% | 30.67% |
| greedy subset | 205.82 | +16.39% | 0.00% |

这说明 top-L 是强先验，但 AP 子集本身仍有约 16% 的状态相关提升空间；
随机子集与可行上界之间的巨大差距也为学习曲线提供了真实的可学习区间。

## 3. AP-CPU 前传有限队列

旧实现只比较当前时隙负载与容量，超限时把该 AP 服务的 UE 速率按比例裁切；
它没有跨时隙状态，因而不能表示短暂突发被缓冲吸收、持续过载形成积压，以及
有限缓冲最终溢出。新实现为每个 AP 维护一个聚合流体队列：

```text
A_m(t)     = slot_dt * sum_{k served by m} R_k(t)
S_m        = slot_dt * C_m
Qmax_m     = buffer_s * C_m
Q_total    = Q_m(t) + A_m(t)
D_m        = min(Q_total, S_m)
Q_m(t+1)   = min(Q_total - D_m, Qmax_m)
drop_m(t)  = max(Q_total - D_m - Qmax_m, 0)
```

历史和当前数据按比例共享 AP 前传服务。UE 当前有效速率按当前到达数据实际获得
的服务比例缩放；缓冲溢出不计入有效吞吐，队列溢出通过 QoS 和速率自然惩罚，
不增加独立奖励项。模型只保留有限队列所需的状态和诊断，不加入随机到达、
多优先级队列或独立时延约束。

主实验参数：

```text
fronthaul_queue = 1
fronthaul_buffer_s = 0.5
C_m ~ Uniform(550, 650) Mbps
backhaul_jt_model = bottleneck
```

`evaluate_current()` 的反事实评估使用冻结队列，不推进时间；只有环境
`step()` 会更新队列。评测 JSON 和结果包同时记录队列占用、丢弃率、服务
利用率和近似排队时延。

## 4. Actor 观测

Actor 使用 1 个 UE 自身 token 和最强 `Kmax=5` 个候选 AP token。开启空间特征
但关闭前传队列时每个 token 为 6 维；开启前传队列后为 7 维。

自身 token：

```text
[log-SINR 归一化,
 当前有效速率 / 速率目标尺度,
 (当前有效速率 - R_req) / R_req,
 R_req 归一化,
 UE x / map 归一化,
 UE y / map 归一化]
```

每个候选 AP token：

```text
[相对最强 AP 的链路增益 dB / 30,
 offered_load / backhaul_capacity 的对数归一化,
 AP 回程有效缩放,
 是否属于当前服务/协作集合,
 AP x / map 归一化,
 AP y / map 归一化,
 AP 前传队列占用 / 有限队列容量]
```

链路增益使用相对量而不是绝对路径损耗，避免地图尺度改变后输入分布漂移；
负载比使用 `log2(1+load/capacity)/2` 压缩重尾；所有 token 特征均有界。
只保留 top-5 AP 是因为动作的协作簇最大为 5，30 个 AP 全量 token 会引入
大量不参与动作选择的远距离 AP。

默认使用4帧历史：

```text
队列关闭且无坐标：6 token x 4 dim = 24 dim
队列关闭 + 坐标：6 token x 6 dim = 36 dim
队列开启 + 坐标：6 token x 7 dim = 42 dim
队列开启 + 坐标 + 4帧：42 x 4 = 168 dim
```

UE token 的队列特征取当前服务簇内最严重的归一化队列占用。AP token 使用
该 AP 自己的队列占用；因此“哪个 AP 即将拥塞”不会被服务簇平均值掩盖。

当前帧位于最后一组token，AP子集打分只使用当前帧；Transformer的池化表示
可以同时使用四帧历史。关闭空间特征和历史可分别设置
`OBS_INCLUDE_XY=0`、`OBS_FRAMES=1`。

## 5. AP 间距与最低速率

AP 最小间距是场景生成硬约束。随机部署完成后会检查两两距离，不满足时直接
报错，不会静默回退到近距离布局。

40 个 AP 布局种子的统计：

| 最小间距 | 最强 5 AP 中存在 `d<25m` | 最强 5 AP 中存在 `d<50m` | top-5 AP 最近距离 p05 |
|---:|---:|---:|---:|
| 0 m | 9.33% | 36.42% | 17.02 m |
| 25 m | 0% | 31.42% | 30.02 m |
| 50 m | 0% | 0% | 52.79 m |
| 75 m | 0% | 0% | 77.59 m |
| 100 m | 0% | 0% | 101.29 m |

主实验取 `AP_min=50m`。它消除了 50 m 内的近似共址 AP，同时仍允许 30 AP
在 1 km x 1 km 内随机部署，不会把网络变成规则网格。

最低速率采用每时隙软约束：

```text
R_k(t) >= R_min = 40 Mbps
```

训练奖励为：

```text
r(t) = mean_k R_k(t)/W
       - lambda_qos * mean_k[
           1(R_k < R_min)
           + max(R_min-R_k,0)/R_min
         ]
```

默认 `lambda_qos=20`。软约束不是把违约行为直接屏蔽，因为联合动作下某个 UE
的动作会通过干扰和回程影响其他 UE；固定动作屏蔽可能产生空动作集。训练过程
同时记录 QoS 违约率、平均速率缺口和每 UE 的 5% 分位速率，评测时不掩盖约束
失败。

40 Mbps 不是论文参数，而是由当前可达速率分布确定的：

| 固定动作 | 20 Mbps 违约率 | 40 Mbps 违约率 | 60 Mbps 违约率 |
|---|---:|---:|---:|
| CB L=5 | 0.05% | 3.98% | 15.43% |
| JT L=2 | 1.02% | 11.98% | 28.67% |
| NC | 0.03% | 8.21% | 34.67% |

论文的 2-20 Mbps 在当前环境中几乎不约束动作。40 Mbps 能明显区分不同动作，
但并非不可达；后续应通过约束满足率而不是奖励数值本身来调 `lambda_qos`。

## 6. 论文事实与簇大小筛选

论文式 (11) 和 (13) 的形式是：

```text
CB: L = 2,3,...,Lmax
JT: L = 2,3,...,Lmax
```

论文仿真取 `Lmax=5`。原始项目代码则在代码中写死 `L=3,4,5`。因此本实验先在
非评测种子上测试 `L=2..6`，不预设 `3,4,5`。

主工作点：

```text
30 UE, 30 AP, 4 antennas/AP, 1 km x 1 km
10 m/s, decision interval 0.5 s
C_m in [550,650] Mbps
AP-CPU finite queue = 0.5 s x C_m
AP_min = 50 m
R_min = 40 Mbps
path loss: original three-slope model
channel aging: off
```

当前有限队列模型在非评测种子 101–103 上的固定 top-L 动作校准如下。该表只
用于检查环境是否退化，不作为最终评测结果：

| 动作 | 系统 SE (bps/Hz) | QoS 违约率 | 前传丢弃率 |
|---|---:|---:|---:|
| JT L=2 | 150.9 | 4.8% | 2.1% |
| JT L=3 | 112.1 | 15.3% | 8.6% |
| JT L=5 | 68.1 | 46.2% | 25.8% |
| CB L=2 | 127.1 | 3.2% | 0.0% |
| CB L=3 | 142.5 | 0.9% | 0.0% |
| CB L=5 | 160.8 | 0.2% | 0.0% |
| NC | 114.7 | 3.5% | 0.0% |

这说明：

- JT 的前传瓶颈要求小簇，`L=2` 明显优于 L3/L5；
- CB 的协作抑制收益要求大簇，`L=5` 明显优于小簇；
- 固定 `CB-L5` 与 `JT-L2` 的 SE 差约 6%，不再像 400–500 Mbps 下超过 20%；
- 五条校准轨迹的逐状态全局最优中，`JT-L2` 为 284/500，`CB-L5` 为
  207/500，另出现 9 次 `JT-L3`，说明模式选择不是固定单选；
- `L=6` 不属于论文主仿真范围，虽然早期无间距诊断时更高，但不进入主实验；
- 在 `L=2..5` 内，非评测种子选择了 `{2,3,5}`，删除 `L=4`。

独立评测种子上的覆盖比较：

| 大小集合 | 最佳固定动作 | 固定动作 Mbps | 全局逐步 oracle Mbps | 可行逐 UE 贪心混合 Mbps |
|---|---|---:|---:|---:|
| 2,3,5 | CB L=5 | 3014.5 | 3020.1 | 3232.0 |
| 3,4,5 | CB L=5 | 3014.5 | 3014.5 | 3140.9 |
| 2,3,4,5 | CB L=5 | 3014.5 | 3020.1 | 3236.7 |

`{2,3,5}` 比最佳固定动作多 `217.5 Mbps`，约 `7.2%`。可行的逐 UE 贪心混合
动作使用：

```text
JT / CB / NC = 41.3% / 54.1% / 4.6%
L=2 / L=3 / L=5 = 33.8% / 13.2% / 52.9%
```

保留第四个大小 `L=4` 只把可行贪心混合吞吐从 `3232.0` 提高到 `3236.7 Mbps`，
增益约 `0.15%`，所以主实验使用三个大小，减少一个动作分量的取值。

以上为历史筛查结论，筛查脚本与中间结果已归档，不在当前精简工程中。

## 7. JT 回程缩放 A/B

原始实现把 JT UE 在多个服务 AP 上的缩放因子连乘：

```text
scale_k = product_m(scale_k,m)
```

新增对照模型取最差回程瓶颈：

```text
scale_k = min_m(scale_k,m)
```

相同 93k 评测轨迹上的固定动作结果：

| 模型 | 动作 | 系统 SE | 回程有效比例 | QoS 违约率 |
|---|---|---:|---:|---:|
| product | JT L=2 | 130.29 | 0.738 | 13.42% |
| bottleneck | JT L=2 | 135.98 | 0.772 | 6.64% |
| product | JT L=3 | 82.57 | 0.411 | 46.19% |
| bottleneck | JT L=3 | 109.66 | 0.550 | 15.30% |
| product | JT L=5 | 12.41 | 0.053 | 95.56% |
| bottleneck | JT L=5 | 70.97 | 0.310 | 42.80% |
| product/bottleneck | CB L=5 | 150.52 | 0.961 | 4.12% |

结论：

- 原实现确实对多 AP JT 进行了过强的复合降速；
- 改成瓶颈模型后，JT L=2 的 QoS 违约率减少约一半，JT L=3 的 SE 提高约 33%；
- CB/NC 完全不受影响，说明 A/B 隔离有效；
- CB L=5 仍是单一固定动作最优，但相对 JT L=2 的优势由约 15.5% 缩小到约 9.7%；
- 逐 UE 最优动作中 JT 占比从 41.0% 提高到 43.0%，`L=2/3` 使用占比也提高。

最终默认采用新回程模型：

```text
backhaul_jt_model = bottleneck
```

回程容量扫描结果：

| C_m 范围 | 固定最优动作 | 最优 SE | CB-JT 差距 |
|---|---|---:|---:|
| 400–500 Mbps | CB L=5 | 150.52 | +14.54 |
| 500–700 Mbps | JT L=2 | 155.25 | -0.20 |
| 700–1000 Mbps | JT L=3 | 170.85 | -14.33 |
| 1000–1200 Mbps | JT L=3 | 188.48 | -31.92 |

以上扫描属于旧的“瞬时容量裁切”模型。有限队列加入后，`400–500 Mbps`
会过度压制 JT，因此当前主实验改用 `550–650 Mbps` 和 `0.5 s` 缓冲；
旧扫描数值只保留为历史对照，不能与当前结果直接合并。

该历史 A/B 的脚本和逐样本结果已归档，不在当前精简工程中。

## 8. 信道老化结论

当前 Jakes 块平均模型使用 200 ms 通信块、10 个导频 slot。10 m/s 时
`rho2` 约为 `0.032`，即相干增益损失约 15 dB。

同一轨迹上的 A/B：

| 设置 | 最佳固定动作 | 吞吐 Mbps | 模式结论 |
|---|---|---:|---|
| aging off | CB L=5 | 3014.5 | CB/JT/NC 均有作用 |
| aging on | JT L=3 | 1219.3 | JT 约占 99.7%，CB/NC 几乎消失 |

因此信道老化不是“越真实越好”就可以直接塞进主实验：

- 原始论文没有建立 UE 移动导致的老化，也没有给出这里的块长、导频比例；
- 当前老化参数会把模式选择问题压成接近固定 JT 的退化问题；
- 主实验默认关闭老化，保证移动性仍是唯一环境扩展；
- 论文中应另列“移动 + 信道老化”鲁棒性实验，而不是用它来制造算法差异。

## 9. 时间平均 SE 目标

主实验不再照搬论文瞬时奖励。单步系统频谱效率定义为：

```text
SE_tot(t) = sum_k R_k(t) / W
```

最终优化目标是一段移动轨迹上的时间平均：

```text
J = (1/T) * sum_{t=1..T} SE_tot(t)
```

训练时每步使用平均 UE-SE 作为共享团队奖励：

```text
r(t) = (1/K) * sum_k R_k(t) / W = SE_tot(t) / K
```

因为 UE 数 `K` 固定，单步奖励与系统总 SE 具有相同的最优策略。奖励量级从
数千 Mbps 降为约 5 到 10 bps/Hz，也更适合 PPO 数值训练。

训练进度显示会跟随奖励模式变化：`mean_se`/`team_sum` 显示对应的 SE，
`mean_rate` 显示平均 UE 速率，`team_rate` 显示 `SysRate` 系统总速率（Mbps）。
因此系统速率实验不会再以 SE 作为终端进度主指标；SE 仍会保存在
`metrics_history.json` 和评测结果中，供最终论文表格使用。

窗口长度筛选：

| T 步 | 时间 | UE 位移 | CB 窗口内 SE 变异 | JT 窗口内 SE 变异 | 每 UE 预期簇变化 |
|---:|---:|---:|---:|---:|---:|
| 20 | 10 s | 100 m | 6.31% | 10.79% | 2.1 |
| 50 | 25 s | 250 m | 5.21% | 9.60% | 5.2 |
| 100 | 50 s | 500 m | 3.71% | 6.40% | 10.4 |
| 200 | 100 s | 1000 m | 1.78% | 3.42% | 20.8 |

主实验取 `T=100` 个决策步，即 `50 s`、10 m/s 下约 `500 m` 位移：

- 每个 UE 平均经历约 7 到 10 次协作簇变化，不是短窗口下的一两个偶发事件；
- CB 和 JT 的窗口内 SE 变异分别约 4% 和 6%，保留移动轨迹差异；
- 相比 `T=200`，100 步显著缩短 PPO 的信用分配距离和训练时长；
- 相比 `T=20/50`，又不会只优化一个局部位置。

默认 `gamma=0.99`，有效记忆长度约为 `1/(1-gamma)=100` 个决策步，与
`objective_horizon=100` 对齐。训练代码要求 `mean_se` 模式下
`episode_length == objective_horizon`。

论文式 (21) 的差分奖励仍保留为 `--reward_mode paper` 的参考消融，但不再是
主实验目标。

## 10. 两种学习算法

比较仍是：

```text
MAPPO + MLP Actor
MAPPO + Transformer Actor
```

二者共享相同的移动环境、输入信息、动作空间、时间平均 SE 奖励、训练步数、
随机种子和评测协议。内部表示与动作头允许不同；公平性是控制问题与实验预算，
而不是强行让两种算法拥有相同的神经网络。

### 10.1 两种架构比较口径

`ARCH_PROFILE=controlled` 保留此前结果所用的受控消融：

```text
MLP：token-aware 输入归一化 + 共享 subset_conditional 动作头
TF ：Transformer token 表示 + 同一个 subset_conditional 动作头
```

该口径只改特征提取骨干，适合回答“自注意力本身有没有贡献”。

`ARCH_PROFILE=native` 是后续主实验口径：

```text
MLP-native：
  168 维扁平观测
  -> 全连接 Actor
  -> mode(3) / cluster-size(4) / AP-member(5) 三个分支
  -> 合法子集映射到 53 个联合动作

TF-native：
  24 个 7 维 token -> Transformer -> 关系型 subset 动作头
```

MLP-native 不包含 token-level self-attention。相比直接输出 53 类动作，
factorized 动作头仍然完全由扁平观测和全连接层实现，只是显式利用环境
已经定义的模式、簇大小和 AP 子集结构。

`ARCH_PROFILE=direct` 保留前一版 MLP“扁平观测 -> 直接输出 53 个动作”的
结构，用于动作头消融；TF 部分继续使用 relational subset head。

TF-native 的动作头包含两条注意力路径：

- 当前 UE token 作为 query，对所有历史 UE/AP token 做 attention pooling，
  产生 mode 和 cluster-size 上下文；
- 当前帧候选 AP 之间做 cross-AP attention，再与 UE 上下文融合后计算 AP
  utility，因此一个 AP 的得分会考虑其他候选 AP 是否互补或冗余。

两种 native 架构仍使用相同的观测信息、AP 候选集合、合法动作掩码、奖励、
训练预算、seed 和调参预算。后续论文主表使用 native；controlled 作为
“只替换骨干”的机制消融。

运行 native 对比：

```bash
ARCH_PROFILE=native \
TF_RELATION_DIM=32 \
SEEDS="2,3,4,5,6,7,8" \
STEPS=400000 \
EPISODE_LENGTH=100 \
SPEED=15 \
NENV=16 \
VEC_ENV=subproc \
REWARD_MODE=team_rate \
QOS_PENALTY_WEIGHT=0 \
SELECT_BEST_CHECKPOINT=0 \
SAVE_CHECKPOINT_SNAPSHOTS=0 \
EVAL_SEED=93000 \
EVAL_EPISODES=20 \
PLOT_METRIC=rate \
TAG=fronthaul_queue_native_rate_400k \
bash run_iicms_comparison.sh
```

复现旧结果时显式设置 `ARCH_PROFILE=controlled`。主对比采用完整架构：

```text
MAPPO-MLP：MLP Actor + MLP Critic
MAPPO-TF： Transformer Actor + Transformer Critic
```

Transformer Critic 将30个 UE 的168维观测视为30个 token，在 UE 维建模
回程容量和频谱效率的相互耦合。`TF Actor + MLP Critic` 保留为 Actor
消融，用于单独判断 Transformer Actor 的贡献。

两个算法的训练超参数可以独立设置。公平性要求使用相同的搜索范围、调参预算、
选择种子和最终评测种子，而不是强制共用同一个学习率。可用参数包括：

```text
MLP_LR / TF_LR
MLP_CRITIC_LR / TF_CRITIC_LR
MLP_ENTROPY_COEF / TF_ENTROPY_COEF
MLP_HIDDEN / TF_HIDDEN
TF_D_MODEL
```

CLS汇总实验已经证明会降低确定性策略性能，不再作为默认配置。默认恢复：

```text
TF_POOL = mean
TF_USE_CLS = 0
TF_CRITIC_BACKBONE = transformer
```

当前阶段冻结环境、168维观测、53动作空间、奖励和前传配置，只测试
Transformer Critic。复现旧版 Actor-only TF：

```bash
TF_CRITIC_BACKBONE=mlp \
TF_POOL=mean TF_USE_CLS=0 \
bash run_iicms_comparison.sh
```

MLP Actor 的输入也按结构化 token 分别做 LayerNorm，即 UE token 和每个
AP token 独立归一化，不再对拼接后的 168 维向量整体归一化；中心化 Critic
仍使用完整全局状态。该设置由 `actor_token_norm=true` 控制，是调用
`disable_actor_token_norm` 可关闭的消融。

MLP 当前默认调参工作点：

```text
MLP_LR = 2e-4
MLP_CRITIC_LR = 1e-4
MLP_ENTROPY_COEF = 0.005
MLP_HIDDEN = 192
```

这是短训筛选后的候选值，不是最终结论；正式比较前仍需与其他候选做相同预算的
validation 调参。

例如分别调参：

```bash
MLP_LR=2e-4 \
MLP_ENTROPY_COEF=0.005 \
MLP_HIDDEN=192 \
TF_LR=5e-5 \
TF_ENTROPY_COEF=0.003 \
TF_HIDDEN=128 \
TF_D_MODEL=64 \
SEEDS="1" STEPS=20000 EPISODE_LENGTH=100 NENV=4 \
TAG=iicms_independent_hparams \
bash run_iicms_comparison.sh
```

Windows：

```powershell
.\run_iicms_comparison.ps1 `
  -MlpLr 2e-4 -MlpEntropyCoef 0.005 -MlpHidden 192 `
  -TfLr 5e-5 -TfEntropyCoef 0.003 -TfHidden 128 -TfDModel 64 `
  -Steps 20000 -Episodes 100 -Envs 4 -Cuda 1 `
  -Tag iicms_independent_hparams
```

调参时使用 validation 种子，不触碰最终 test 种子：

```bash
python evaluate_iicms.py \
  --run_dir results/IICMSMobility/PureMobility/mappo/<run> \
  --seed_start 92000 --episodes 20 --horizon 100 \
  --phase validation --checkpoint best \
  --out results_validation
```

默认使用确定性 argmax。需要检查策略分布本身时可增加 `--stochastic`；
这与论文主表的确定性执行结果应分开报告。

91000–91019用于checkpoint选择和训练中验证，92000–92019用于算法级超参数选择，93000–93019
为最终test。`evaluate_iicms.py --phase test`会读取`checkpoint_selection.json`，
发现test与checkpoint选择种子重叠时直接报错。每种算法应使用相同数量的调参组合
和相同的validation/test轨迹，不能用不同的调参预算。

## 11. Validation checkpoint 选择

训练脚本默认每25000个环境步在固定validation种子91000–91019上执行20个
确定性episode，并同时写出：

```text
logs/validation_curve.csv
logs/validation_history.json
logs/validation_convergence.json
```

验证数据由训练、checkpoint选择和最终评估共用同一个episode执行核心。每个
validation episode都会重新配置环境seed，因此AP部署、回程容量、阴影、UE起点
和RWP路径在MLP与TF之间一致。`validation_convergence.json`用最近3个点的
固定场景斜率估计当前增长趋势；只有斜率95%上界对应的系统速率增长不超过
0.5%/100k步时，才报告`plateau_candidate`。否则保持`continue_training`。
该判断同时参考P05和QoS，不能用随机训练回报代替。

实验脚本默认在每个`save_interval`保存带步数的actor/critic快照。训练结束后，
`select_best_checkpoint.py`会在固定validation种子91000–91019上执行确定性
评测。默认`BEST_CHECKPOINT_METRIC=auto`：

- 纯速率训练且`qos_penalty_weight=0`时选择`system_se`；
- 配置正QoS惩罚时选择`qos_constrained`。

如果显式使用QoS约束规则，采用：

```text
eligible = QoS违约率 <= max_qos_violation_rate
best = argmax mean_UE_SE among eligible
```

默认 `max_qos_violation_rate=0.005`。如果没有 checkpoint 满足约束，则先选
违约率最低的候选，再在这些候选中选择 SE 最大者。`qos_aware` 加权评分保留
为消融。纯系统速率实验不应使用该后置约束规则。checkpoint选择默认使用20个
episode；5个episode只适合早期筛选，不能用于最终选择。

每次validation episode都会用该episode的seed重新配置环境和场景，而不是沿用
训练的`seed`。因此AP部署、回程容量、阴影、UE起点和RWP路径在MLP与TF之间
完全一致。最终test也采用相同规则，并在JSON中记录每次episode初始场景的
SHA256摘要；摘要或`test_protocol_id`不一致时不能视为同一测试轨迹。

选择结果写入`checkpoint_selection.json`，同时保留：

```text
models/actor_final.pt
models/critic_final.pt
models/actor_best.pt
models/critic_best.pt
```

最终`actor.pt/critic.pt`会被替换为validation最佳模型。最终test评测仍使用
未参与选择的93000–93019种子。

先用一个种子跑800k收敛探针，不消耗93000–93019最终test种子：

```bash
SEEDS=1 \
STEPS=800000 \
EPISODE_LENGTH=100 \
SPEED=15 \
NENV=16 \
VEC_ENV=subproc \
ALGOS=both \
ARCH_PROFILE=native \
REWARD_MODE=team_rate \
QOS_PENALTY_WEIGHT=0 \
TRAINING_VALIDATION=1 \
VALIDATION_INTERVAL_STEPS=25000 \
VALIDATION_EPISODES=20 \
VALIDATION_SEED=91000 \
VALIDATION_HORIZON=100 \
VALIDATION_PLATEAU_RELATIVE_GAIN=0.005 \
VALIDATION_WINDOW=3 \
SELECT_BEST_CHECKPOINT=0 \
SAVE_CHECKPOINT_SNAPSHOTS=0 \
AUTO_EVAL=0 \
PLOT_RESULTS=1 \
PLOT_METRIC=rate \
TAG=rate_convergence_pilot_800k \
bash run_iicms_comparison.sh
```

探针结束后查看
`results/IICMSMobility/PureMobility/mappo/rate_convergence_pilot_800k_figures/rate_convergence_pilot_800k_validation_curves.png`
以及每个run的`logs/validation_convergence.json`。若平台期出现在`K`步附近，
正式实验统一使用`STEPS=K`并重新跑完整seed集合，避免探针种子获得额外训练预算。
例如平台期为600k时，再运行：

```bash
SEEDS="1,2,3,4,5,6" \
STEPS=600000 \
EPISODE_LENGTH=100 \
SPEED=15 \
NENV=16 \
VEC_ENV=subproc \
ALGOS=both \
ARCH_PROFILE=native \
REWARD_MODE=team_rate \
QOS_PENALTY_WEIGHT=0 \
TRAINING_VALIDATION=1 \
VALIDATION_INTERVAL_STEPS=25000 \
VALIDATION_EPISODES=20 \
VALIDATION_SEED=91000 \
VALIDATION_HORIZON=100 \
SELECT_BEST_CHECKPOINT=1 \
BEST_CHECKPOINT_METRIC=auto \
BEST_CHECKPOINT_EPISODES=20 \
BEST_CHECKPOINT_SEED=91000 \
AUTO_EVAL=1 \
EVAL_SEED=93000 \
EVAL_EPISODES=20 \
PLOT_RESULTS=1 \
PLOT_METRIC=rate \
TAG=rate_main_600k \
bash run_iicms_comparison.sh
```

Windows 单种子探针对应命令：

```powershell
.\run_iicms_comparison.ps1 `
  -Seeds 1 -Steps 800000 -Episodes 100 -Envs 16 `
  -Algos both -ArchitectureProfile native `
  -RewardMode team_rate -QosPenaltyWeight 0 `
  -TrainingValidation 1 -ValidationIntervalSteps 25000 `
  -ValidationEpisodes 20 -ValidationSeed 91000 -ValidationHorizon 100 `
  -SelectBestCheckpoint 0 -SaveCheckpointSnapshots 0 `
  -AutoEval 0 -PlotResults 1 -PlotMetric rate `
  -Tag rate_convergence_pilot_800k
```

训练和超参数搜索可以只运行一个算法：

```bash
ALGOS=tf MLP_LR=2e-4 MLP_ENTROPY_COEF=0.005 \
TF_LR=2e-4 TF_ENTROPY_COEF=0.005 \
SEEDS="1,2,3" STEPS=100000 EPISODE_LENGTH=100 NENV=8 \
EVAL_SEED=92000 EVAL_PHASE=validation \
TAG=tf_hparam_candidate \
bash run_iicms_comparison.sh
```

`ALGOS=mlp`和`ALGOS=tf`分别用于独立调参，`ALGOS=both`表示现有
MLP-native与Transformer-native两种策略，`ALGOS=orig`用于原论文风格
MAPPO第三条学习策略，`ALGOS=all`一次运行三种策略。
每种算法可以拥有自己的学习率和熵系数，但两者必须使用相同数量的候选组合、
训练预算、validation种子和最终test种子。

### 11.1 原论文风格 MAPPO 对比策略

`ALGOS=orig`对应 `original_mappo`：

- 30个UE各自拥有独立的MLP Actor和Critic，不共享参数；
- Actor只输入本UE的局部观测；
- Critic输入全部UE局部观测拼接成的中心化状态；
- Actor保留论文中的两个分类分支：`JT/CB/NC` 与配置的簇大小；
- 模式与大小分支的联合对数概率用于PPO更新；
- 为匹配当前环境动作，仅执行每种模式/大小对应的top-L簇；
- 环境、系统速率奖励、训练步数、训练种子和 `93000–93019` test种子
  与MLP-native、Transformer-native保持一致。

这里复现的是论文的独立智能体CTDE结构和模式×簇大小动作分解，不复制原实现
中已经修正的PPO问题。当前版本对两个分类分支使用联合log-probability，
并保证熵项能够向两个分支传播梯度；因此结果中会同时记录
`original_mappo=true`、`share_policy=false`和
`joint_action_head=paper_factorized`，便于区分。

在现有两种400k主实验完成后，用相同TAG追加第三策略：

```bash
SEEDS="1,2,3,4,5,6" \
STEPS=400000 \
EPISODE_LENGTH=100 \
SPEED=15 \
NENV=16 \
VEC_ENV=subproc \
ALGOS=orig \
ARCH_PROFILE=native \
CLUSTER_ACTION_MODE=subset \
REWARD_MODE=team_rate \
QOS_PENALTY_WEIGHT=0 \
MIN_UE_SE_BPS_HZ=0.5 \
SELECT_BEST_CHECKPOINT=0 \
SAVE_CHECKPOINT_SNAPSHOTS=0 \
AUTO_EVAL=1 \
PLOT_METRIC=rate \
PACKAGE_RESULTS=1 \
TAG=fronthaul_queue_native_rate_400k \
bash run_iicms_comparison.sh
```

原论文风格策略默认使用 `PAPER_HIDDEN=64`、`PAPER_LR=5e-5`、
`PAPER_CRITIC_LR=5e-5`、`PAPER_ENTROPY_COEF=0.01` 和
`PAPER_PPO_EPOCH=15`。这些是该策略自身的超参数，不要求与MLP-native或
Transformer-native相同；环境、奖励、训练预算、随机种子和评测协议完全相同。

三条学习曲线可单独汇总：

```bash
python plot_training_curves.py \
  --tag fronthaul_queue_native_rate_400k \
  --seeds 1,2,3,4,5,6 \
  --families mlp,tf_actor,orig_mappo \
  --metric rate \
  --run latest \
  --out results/IICMSMobility/PureMobility/mappo/fronthaul_queue_native_rate_400k_figures/three_learning_training_curves.png
```

参考论文式训练 reward 收敛图：

```bash
python plot_reward_convergence.py \
  --tag fronthaul_queue_native_rate_400k \
  --seeds 1,2,3,4,5,6 \
  --families mlp,tf_actor,orig_mappo \
  --episode-length 100 \
  --source raw \
  --out results/IICMSMobility/PureMobility/mappo/fronthaul_queue_native_rate_400k_figures/three_learning_reward_convergence.png
```

脚本优先读取每个 run 的`reward_proposed.npy`或`reward_proposed.mat`，按
`episode_length`聚合为 episode 内的每步平均 reward，并画多种子均值和
95%置信带。单种子时使用episode内标准差带。默认`--source raw`要求原始逐步
reward存在，缺失时直接报错。只有显式传入`--source auto`才允许回退到
`curve.csv`，并会在标题和JSON中标记为fallback。

固定模式/RSSI基线与三条学习策略合并绘图时，传：

```bash
FAMILIES=mlp,tf_actor,orig_mappo \
RUN_DIR=results/IICMSMobility/PureMobility/mappo/fronthaul_queue_native_rate_400k_orig_mappo_s1/run1 \
RSSI_THRESHOLDS_FILE=results/IICMSMobility/PureMobility/mappo/fronthaul_queue_native_rate_400k_orig_mappo_s1/run1/rssi_threshold_calibration/selected_rssi_thresholds.env \
bash run_baseline_suite.sh
```

### 11.2 Zero-shot AP 数量扩展

已有 MLP/TF 模型训练时使用30个AP。当前局部观测只包含最强
`max(cluster_sizes)` 个AP，因此模型输入维度不随AP总数改变，可以直接用
同一套权重在 `M={10,20,30,40,50}` 上测评。该实验不重新训练模型：

```bash
python evaluate_ap_count_sweep.py \
  --mlp-tag fronthaul_queue_mlp_factorized_rate_400k \
  --tf-tag fronthaul_queue_native_rate_400k \
  --old-mlp-tag fronthaul_queue_native_rate_400k \
  --mlp-seeds 1,2,3,4,5,6 \
  --old-mlp-seeds 1,2,3,4,5,6 \
  --tf-seeds 1,2,3,4,5,6 \
  --families mlp,old_mlp,tf \
  --ap-counts 10,20,30,40,50 \
  --baselines dc_rssi,sc_rssi,dc_cb_l5,sc_cb_l5,dc_jt_l2,sc_jt_l3,dc_nc,sc_nc \
  --seed-start 93000 \
  --episodes 20 \
  --horizon 100 \
  --reuse-m30 \
  --output-dir analysis_outputs/ap_count_zero_shot
```

脚本自动查找已有RSSI门限文件；M=30时复用已存在且与测评种子、episode数和
horizon一致的确定性结果。输出包括：

```text
analysis_outputs/ap_count_zero_shot/
├── aggregate_results.csv
├── raw_results.csv
├── sweep_manifest.json
├── ap_count_scalability.png
├── ap_count_main_comparison.png
├── ap_count_tf_vs_mlp_advantage.png
└── ap_count_advantage_summary.json
```

`ap_count_scalability.png`保留全部策略。`ap_count_main_comparison.png`
只展示学习策略和两个代表性基线；`ap_count_tf_vs_mlp_advantage.png`
给出TF相对MLP的绝对速率增益、百分比增益、P05增益和QoS差值，并附95%配对
置信区间。
所有策略在同一 `(M, test seed)` 下使用相同初始场景；`sweep_manifest.json`
中的 `scenario_digests_consistent` 用于检查这一点。

### 11.3 Zero-shot 回程容量扩展

参考论文图5，固定30个AP和30个UE，只改变平均每AP回程容量：

```bash
python evaluate_backhaul_capacity_sweep.py \
  --mlp-tag fronthaul_queue_mlp_factorized_rate_400k \
  --old-mlp-tag fronthaul_queue_native_rate_400k \
  --tf-tag fronthaul_queue_native_rate_400k \
  --mlp-seeds 1,2,3,4,5,6 \
  --old-mlp-seeds 1,2,3,4,5,6 \
  --tf-seeds 1,2,3,4,5,6 \
  --families mlp,old_mlp,tf \
  --capacities-gbps 0.30,0.45,0.60,0.75,1.00,1.50 \
  --variation-gbps 0.05 \
  --reference-capacity-gbps 0.60 \
  --baselines dc_rssi,sc_rssi,dc_cb_l5,sc_cb_l5,dc_jt_l2,sc_jt_l3,dc_nc,sc_nc \
  --seed-start 93000 \
  --episodes 20 \
  --horizon 100 \
  --skip-existing \
  --output-dir analysis_outputs/backhaul_capacity_zero_shot
```

每个容量点使用 `C_avg ± 0.05 Gbps` 的每AP容量范围，因此
`C_avg=0.60 Gbps` 与当前主实验的 `550–650 Mbps` 完全一致，并默认复用
该点的已有确定性测评。输出包括全部策略图、代表性策略主图以及按论文图5
风格绘制的“总速率–平均回程容量”图。

## 12. 运行

先运行测试：

```bash
python -m unittest discover -s tests -v
```

AutoDL 短训：

```bash
chmod +x run_iicms_comparison.sh

SEEDS="1" \
STEPS=20000 \
EPISODE_LENGTH=100 \
NENV=4 \
VEC_ENV=subproc \
TAG=iicms_gated_pilot \
bash run_iicms_comparison.sh
```

学习对比脚本默认使用可学习 AP 子集。也可以显式指定动作空间：

```bash
BACKHAUL_JT_MODEL=bottleneck \
CLUSTER_ACTION_MODE=subset \
SEEDS="1" STEPS=100000 EPISODE_LENGTH=100 NENV=4 \
TAG=iicms_gated_100k \
bash run_iicms_comparison.sh
```

确认趋势后再跑：

```bash
SEEDS="1" \
STEPS=200000 \
EPISODE_LENGTH=100 \
NENV=16 \
VEC_ENV=subproc \
TAG=iicms_gated_main \
bash run_iicms_comparison.sh
```

独立“纯系统速率、最终模型”实验：

```bash
SEEDS="2,3,4,5,6,7,8" \
STEPS=400000 \
EPISODE_LENGTH=100 \
SPEED=15 \
NENV=16 \
VEC_ENV=subproc \
REWARD_MODE=team_rate \
QOS_PENALTY_WEIGHT=0 \
SELECT_BEST_CHECKPOINT=0 \
SAVE_CHECKPOINT_SNAPSHOTS=0 \
MIN_UE_SE_BPS_HZ=0.5 \
PLOT_METRIC=rate \
TAG=fronthaul_queue_rateonly_400k \
bash run_iicms_comparison.sh
```

主实验设置下运行 400k 纯系统速率实验，并保持 validation/test 口径：

```bash
SEEDS="2,3,4,5,6,7,8" \
STEPS=400000 \
EPISODE_LENGTH=100 \
SPEED=15 \
NENV=16 \
VEC_ENV=subproc \
REWARD_MODE=team_rate \
QOS_PENALTY_WEIGHT=0 \
MIN_UE_SE_BPS_HZ=0.5 \
SELECT_BEST_CHECKPOINT=1 \
BEST_CHECKPOINT_METRIC=system_se \
BEST_CHECKPOINT_EPISODES=20 \
BEST_CHECKPOINT_SEED=91000 \
EVAL_SEED=93000 \
EVAL_EPISODES=20 \
AUTO_EVAL=1 \
PLOT_METRIC=rate \
PACKAGE_RESULTS=1 \
TAG=fronthaul_queue_main_rate_400k \
bash run_iicms_comparison.sh
```

该实验中：

- 奖励使用系统总速率 `sum_k R_k(t)`，单位为 Mbps；
- 使用与纯速率目标一致的 `system_se` validation checkpoint 选择；
- 训练中每25000步执行20个固定种子的确定性validation；
- `MIN_UE_SE_BPS_HZ` 只记录最低 UE SE 和 floor 违约率，不改变奖励；
- MLP 与 TF 的模型选择和评价口径完全相同。

如果要求“任何 UE 都不得低于 SE floor”，则需要额外的硬安全层或约束 RL；
仅关闭 QoS 奖励并记录 floor 不能提供严格保证。

脚本结束时会自动生成：

```text
review_bundles/<TAG>_review_<timestamp>.zip
```

成功完成且 `PLOT_RESULTS=1` 时，还会自动生成并打包：

```text
results/IICMSMobility/PureMobility/mappo/<TAG>_figures/
├── <TAG>_training_curves.png
├── <TAG>_training_curves.json
├── <TAG>_reward_convergence.png
├── <TAG>_reward_convergence.json
├── <TAG>_validation_curves.png
├── <TAG>_validation_curves.json
├── <TAG>_validation_p05.png
├── <TAG>_validation_qos.png
├── <TAG>_rate_cdf.png
└── <TAG>_rate_cdf.json
```

训练完成后脚本默认自动执行确定性评测和随机评测，种子范围为
`93000–93019`，每种评测20个 episode、每个100步。结果写入对应 run 的：

```text
evaluation_deterministic/run1.json
evaluation_stochastic/run1.json
baseline_fixed_cb_l5_deterministic/run1.json
```

最终确定性评测 JSON 同时保存每个 seed 的 per-UE SE 经验 CDF。对已经完成
评测的多 seed 结果绘制平均 CDF：

```bash
python plot_seed_se_cdf.py \
  --tag fronthaul_queue_speed15 \
  --seeds 2,3,4,5,6,7,8 \
  --families mlp,tf_actor \
  --metric se \
  --run run1 \
  --out results_figures/fronthaul_queue_speed15_se_cdf.png
```

绘图脚本会输出逐 seed CDF、均值、均值95%区间，并标出
`R_req/W`。旧版本生成的 evaluation JSON 不包含 CDF，需要先用当前评测器
重新执行确定性评测。

并随训练日志和指标一起打入review包。可通过`AUTO_EVAL=0`关闭，
或通过 `EVAL_SEED`、`EVAL_EPISODES`、`EVAL_HORIZON` 修改评测设置。
`EVAL_SEED`范围与`BEST_CHECKPOINT_SEED`范围重叠时脚本会直接退出。
固定基线由`BASELINE_ACTION`控制，默认`CB_L5_S00`；`EVAL_BASELINES=0`
可关闭。基线、MLP和TF使用同一环境、同一组test seed和记录在JSON中的同一个
`test_protocol_id`。注意，这里的固定动作是“固定模式/大小、动态候选簇”，
仅作为旧结果诊断，不作为论文中的严格固定模式基线。

### 12.1 静态固定模式与 RSSI 基线

论文对比中的固定模式基线采用 **episode-static UE-centric clustering**：

- 每个 test episode reset 时，根据该 episode 初始大尺度信道确定每个 UE
  的候选 AP 排名；
- 固定该排名和物理服务簇，整个 100-step episode 不再重新排序；
- UE 仍以 15 m/s 移动，信道仍按新位置更新；
- 一个 episode 内没有物理 AP 簇切换，因此 `cluster_change_rate=0`；
- 不同 test seed 可以拥有不同初始 AP 集合。

这不是“所有 episode 共用同一套拓扑分区”的全局静态分区，而是与本项目
移动性公平对比最直接对应的 no-intra-episode-HO 定义。

```bash
RUN_DIR=results/IICMSMobility/PureMobility/mappo/<已训练run>/run1 \
EVAL_SEED=93000 \
EVAL_EPISODES=20 \
EVAL_HORIZON=100 \
RSSI_CLUSTER_SIZE=5 \
RSSI_THRESHOLDS_FILE="${RUN_DIR}/rssi_threshold_calibration/selected_rssi_thresholds.env" \
bash run_baseline_suite.sh
```

该脚本统一评测：

- `SC-NC`；
- `SC-CB-L2/L3/L4/L5`；
- `SC-JT-L2/L3/L4/L5`；
- `DC-NC`；
- `DC-CB-L2/L3/L4/L5`；
- `DC-JT-L2/L3/L4/L5`；
- `SC-RSSI`；
- `DC-RSSI`。

RSSI 基线使用原始项目中的链路强度对比门限：

```text
ratio = best_link / sum(other_links)

0 < ratio <= low       -> JT
low < ratio <= high    -> CB
ratio > high           -> NC
```

`SC-RSSI` 与 `DC-RSSI` 应使用相同的门限；门限只能在 validation seeds
上校准。结果写入 `${RUN_DIR}/baseline_suite/`，并自动生成
`baseline_summary.csv/json`。

先校准阈值：

```bash
RUN_DIR=results/IICMSMobility/PureMobility/mappo/<已训练run>/run1 \
VALIDATION_SEED_START=92000 \
SCREEN_EPISODES=5 \
FINAL_EPISODES=20 \
python tune_rssi_thresholds.py \
  --run_dir "$RUN_DIR" \
  --validation_seed_start "$VALIDATION_SEED_START" \
  --screen_episodes "$SCREEN_EPISODES" \
  --final_episodes "$FINAL_EPISODES" \
  --horizon 100 \
  --cluster_size 5 \
  --high_grid "0.5,1,2,4,8,16,32,64,128,256"
```

脚本先在 5 个 validation episode 上搜索候选门限，再在 20 个 validation
episode 上复测前 5 名。选择目标为 `SC-RSSI` 和 `DC-RSSI` 的平均系统速率，
QoS 违约率用于同分判定。输出：

```text
rssi_threshold_calibration/
├── rssi_threshold_calibration.json
├── rssi_threshold_calibration.csv
└── selected_rssi_thresholds.env
```

如果共享、静态或动态策略最终选择的低/高门限仍等于搜索网格边界，
`rssi_threshold_calibration.json`会写入`boundary_warnings`，终端也会输出
`[WARN]`。此时应扩大对应网格后重新校准，不能把边界值直接当作已收敛最优。

最终 test seeds 不参与门限选择。

`run_baseline_suite.sh` 在 `PLOT_RESULTS=1` 时还会生成策略级 per-UE rate
CDF 对比图。主面板包含 MLP、TF、SC-RSSI 和 DC-RSSI；第二面板对
固定 NC、CB、JT 分别在静态和动态聚类下只展示平均系统速率最高的簇大小
版本。全部簇大小的原始结果仍保留在 `baseline_summary.csv/json` 中。输出位于：

```text
results/IICMSMobility/PureMobility/mappo/<TAG>_figures/
├── <TAG>_strategy_rate_cdf.png
└── <TAG>_strategy_rate_cdf.json
```

也可以单独重新绘图：

```bash
python plot_strategy_cdf.py \
  --tag "$TAG" \
  --seeds 2,3,4,5,6 \
  --families mlp,tf_actor \
  --baseline-suite-dir "$RUN_DIR/baseline_suite" \
  --out-prefix "results/IICMSMobility/PureMobility/mappo/${TAG}_figures/${TAG}_strategy_rate_cdf"
```

结果包默认使用精简模式`PACKAGE_MODE=results`，只保留：

```text
config.json
checkpoint_selection.json
evaluation_deterministic/
evaluation_stochastic/
baseline_fixed_cb_l5_deterministic/
logs/curve.csv
logs/metrics_history.json
logs/summary.json
logs_iicms/*.log
```

中间`actor_step_*`/`critic_step_*`、TensorBoard事件文件和
`reward_proposed.npy`不会进入结果包。需要保留最佳权重时设置
`PACKAGE_MODE=best`；只有明确需要完整复查时才使用`PACKAGE_MODE=full`。
需要把已有的其他实验tag一并装入同一结果包做直接比较时，可使用：

```bash
PACKAGE_INCLUDE_TAGS="old_tag_a,old_tag_b"
```

对已经训练完成、尚未评测的五个种子批量补评测并自动打包：

```bash
TAG=cap450_mlpopt_250k \
SEEDS="1,2,3,4,5" \
EVAL_SEED=93000 \
EVAL_EPISODES=20 \
EVAL_HORIZON=100 \
CHECKPOINT=actor \
PACKAGE_MODE=results \
bash evaluate_existing_runs.sh
```

该脚本要求 `results/IICMSMobility/PureMobility/mappo/` 下已经存在
`${TAG}_mlp_s*` 和 `${TAG}_tf_actor_s*` 的对应checkpoint。新实验默认评测
`run1/models/actor.pt`；如果新训练显式开启了快照并希望选择validation最佳
权重，再把 `CHECKPOINT` 改为 `best`。批量脚本也会在每个run下补跑同一
test轨迹的固定CB-L5。

修改代码或筛选结果后，一键运行测试、更新 `SOURCE_SHA256.json` 并生成与
之前格式相同的工程包：

Windows：

```powershell
.\package_project.ps1
```

Linux / AutoDL：

```bash
bash package_project.sh
```

同时生成时间戳工程包、时间戳结果包，以及稳定的最新版本：

```text
release_bundles/clean_iicms_mobility_gated_v1_latest.zip
release_bundles/clean_iicms_mobility_gated_v1_latest_results.zip
```

## 13. 当前验证状态

- 单元测试覆盖旧模型兼容、队列积压/溢出/排空、冻结反事实评估、reset、
  168 维观测、动作头、评测协议、checkpoint 选择和 SE-CDF；
- MAPPO-MLP 与 MAPPO-Transformer 均完成当前 15 m/s 主配置的 CPU
  冒烟训练；
- 两者都识别 `actor_tokens=24 / token_dim=7`，Actor 输入为 168 维，
  Critic 每个 UE token 为 168 维；
- 学习策略、固定动作、checkpoint 选择和评测结果均能正常写出；
- 当前主配置为 15 m/s、全车辆、100-step episode；保留新 MLP-factorized、
  旧 MLP-flat、TF-native 三组六种子主实验，以及 AP 数量和回程容量扩展结果；
