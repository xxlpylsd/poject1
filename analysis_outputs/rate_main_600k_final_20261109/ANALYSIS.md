# `rate_main_600k` six-seed and RSSI analysis

## Scope

- Training seeds: 1-6 for MLP and TF.
- Training budget: 600k environment steps per run.
- Validation: 20 deterministic episodes per checkpoint on seeds 91000-91019.
- Checkpoint selection: `system_se`, 20 episodes per candidate.
- Final test: seeds 93000-93019, 20 episodes.
- RSSI thresholds were calibrated on seeds 92000-92019.
- The review bundle includes the full fixed-mode baseline suite.

## Convergence

Family-mean system rate:

| Family | 0-200k | 200-400k | 400-600k | Last-200k slope | Last-100k slope |
|---|---:|---:|---:|---:|---:|
| MLP | 2935.1 | 3214.4 | 3298.2 | +16.4 Mbps/100k (0.50%) | +27.0 Mbps/100k (0.81%) |
| TF | 2789.0 | 3355.4 | 3377.7 | +5.0 Mbps/100k (0.15%) | +15.7 Mbps/100k (0.46%) |

TF is effectively converged by 400k. MLP retains a small late improvement, but
the last-200k family mean is exactly at the 0.5%/100k practical threshold.

Selected MLP checkpoints average 540.3k (90.0% of budget). Selected TF
checkpoints average 424.0k (70.7%). TF benefits substantially from selecting
earlier checkpoints. The common 600k budget is defensible for the final
comparison; a strict convergence study could use one 800k MLP pilot, but does
not justify retraining all 12 policies.

The three-point convergence JSON should not be used alone: it still classifies
significant negative slopes as `plateau_candidate`. The full 24-point curves
above are the reliable evidence.

## MLP vs TF on deterministic test

| Metric | MLP | TF | Paired TF-MLP |
|---|---:|---:|---:|
| Total rate | 3358.3 Mbps | 3455.7 Mbps | +97.4 Mbps, 95% CI +51.2 to +143.6 |
| P05 | 57.5 Mbps | 65.7 Mbps | +8.2 Mbps, 95% CI +3.4 to +13.0 |
| QoS violation | 1.79% | 0.84% | -0.95 pp, 95% CI -1.53 to -0.38 |
| Overload | 21.27% | 16.66% | -4.61 pp, CI crosses zero |

TF wins total rate, P05 and QoS on all six training seeds. The deterministic
advantage is now statistically and directionally robust.

## Stochastic execution

MLP is better under sampled actions:

| Metric | MLP | TF |
|---|---:|---:|
| Total rate | 3073.4 Mbps | 3013.1 Mbps |
| P05 | 44.1 Mbps | 41.8 Mbps |
| QoS violation | 4.04% | 4.70% |

TF loses 60.3 Mbps under stochastic execution (95% CI -113.8 to -6.8). The
deterministic advantage therefore does not transfer to random action sampling.

## Baselines

On the common test panel:

| Policy | Rate | P05 | QoS | Overload |
|---|---:|---:|---:|---:|
| MLP | 3358.3 | 57.5 | 1.79% | 21.27% |
| TF | 3455.7 | 65.7 | 0.84% | 16.66% |
| DC-RSSI extended | 3145.3 | 52.4 | 1.22% | 5.44% |
| DC-CB-L5 | 3145.3 | 52.4 | 1.22% | 5.44% |
| DC-JT-L2 | 2830.2 | 37.9 | 6.58% | 46.42% |
| SC-RSSI | 1418.0 | 1.9 | 54.86% | 24.80% |

Relative to DC-RSSI:

- MLP: +213.0 Mbps (+6.77%), P05 +5.04 Mbps, QoS 0.57 pp worse;
- TF: +310.4 Mbps (+9.87%), P05 +13.24 Mbps, QoS 0.39 pp better.

TF also improves overload relative to both learned MLP and DC-RSSI. Static
RSSI is not a competitive mobility baseline because clustering is frozen
inside each episode; DC-RSSI is the relevant heuristic comparison.

At the 40 Mbps requirement, the empirical CDF violation fractions are:

- MLP: 1.788%;
- TF: 0.838%;
- DC-RSSI extended: 1.213%.

## RSSI calibration caveat

Extended-grid selected thresholds:

- SC-RSSI: low 0.1, high 64.0;
- DC-RSSI: low 0.1, high 128.0.

The high-threshold grid now extends to 256. No selected threshold lies on a
search boundary and `boundary_warnings` is empty. Extending DC-RSSI from 64 to
128 changes validation dynamically-clustered rate by only about 0.06 Mbps and
changes the independent test result from 3148.7 to 3145.3 Mbps, so the original
comparison conclusion is unchanged.

## Action distribution

Deterministic behavior:

| Family | Top-1 share | Effective actions | Cluster change | JT/CB/NC |
|---|---:|---:|---:|---:|
| MLP | 46.3% | 3.65 | 34.4% | 59.2/40.3/0.5% |
| TF | 37.3% | 6.68 | 37.8% | 66.4/33.5/0.2% |

Neither policy collapses. MLP still uses a small set of dominant joint actions,
while TF changes AP membership more diversely. This is consistent with TF's
higher deterministic P05 and rate, but is not proof of causality.

## Conclusion

The six-seed results are usable for the deterministic main comparison:

- TF has a robust 2.9% total-rate advantage;
- TF has a 14% P05 advantage and materially lower QoS violations;
- TF beats the relevant extended DC-RSSI baseline by about 9.9%;
- the advantage must be described as deterministic and tail-performance
  related, not as a general sampled-policy advantage.

The extended RSSI search is complete and no threshold remains on a grid
boundary. Keep the deterministic result as the primary result. No common
retraining to 800k is required. The only pending artifact is the raw-reward
convergence plot, which must be generated on the AutoDL directories that still
contain `reward_proposed.npy`.
