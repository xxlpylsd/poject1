param(
    [int[]]$Seeds = @(1),
    [int]$Steps = 100000,
    [int]$Episodes = 100,
    [int]$Envs = 4,
    [int]$Cuda = 1,
    [int]$AutoEval = 1,
    [string]$PackageMode = "results",
    [string[]]$PackageIncludeTags = @(),
    [string]$Algos = "both",
    [string]$RewardMode = "mean_se",
    [int]$SaveCheckpointSnapshots = 1,
    [int]$SelectBestCheckpoint = 1,
    [string]$BestCheckpointMetric = "qos_constrained",
    [double]$BestCheckpointMaxQos = 0.005,
    [int]$BestCheckpointEpisodes = 5,
    [int]$BestCheckpointSeed = 91000,
    [int]$EvalSeed = 93000,
    [int]$EvalEpisodes = 20,
    [int]$EvalHorizon = 100,
    [string]$EvalPhase = "test",
    [int]$EvalBaselines = 1,
    [int]$PlotResults = 1,
    [string]$PlotMetric = "se",
    [string]$FigureDir = "",
    [int]$SaveInterval = 10,
    [string]$BaselineAction = "CB_L5_S00",
    [string]$BaselineLabel = "fixed_cb_l5",
    [int]$ObsIncludeXy = 1,
    [int]$ObsFrames = 4,
    [double]$QosPenaltyWeight = 20.0,
    [double]$MinUeSeBpsHz = 0.0,
    [double]$Speed = 15.0,
    [int]$FronthaulQueue = 1,
    [double]$FronthaulBufferS = 0.5,
    [string]$Python = "D:\Anaconda\envs\DL\python.exe",
    [string]$Tag = "iicms_mobility",
    [string]$JointActionHead = "subset_conditional",
    [ValidateSet("controlled", "native", "direct")]
    [string]$ArchitectureProfile = "controlled",
    [int]$TfRelationDim = 32,
    [string]$ClusterActionMode = "subset",
    [double]$MlpLr = 2e-4,
    [double]$MlpCriticLr = 1e-4,
    [double]$MlpEntropyCoef = 0.005,
    [int]$MlpHidden = 192,
    [string]$MlpActorTemporalContext = "history_mean",
    [double]$TfLr = 1e-4,
    [double]$TfCriticLr = 1e-4,
    [double]$TfEntropyCoef = 0.01,
    [int]$TfHidden = 128,
    [int]$TfDModel = 64,
    [string]$MlpCriticBackbone = "mlp",
    [string]$TfCriticBackbone = "transformer",
    [string]$TfPool = "mean",
    [string]$TfActorTemporalContext = "history_mean",
    [int]$TfUseCls = 0,
    [double]$PaperLr = 5e-5,
    [double]$PaperCriticLr = 5e-5,
    [double]$PaperEntropyCoef = 0.01,
    [int]$PaperHidden = 64,
    [int]$PaperPpoEpoch = 15
)

$ErrorActionPreference = "Stop"
$env:OMP_NUM_THREADS = "1"
$env:MKL_NUM_THREADS = "1"
$env:OPENBLAS_NUM_THREADS = "1"
Write-Host "[architecture] profile=$ArchitectureProfile"

$selectionEnd = $BestCheckpointSeed + $BestCheckpointEpisodes - 1
$evaluationEnd = $EvalSeed + $EvalEpisodes - 1
if (
    $AutoEval -eq 1 -and
    $SelectBestCheckpoint -eq 1 -and
    $EvalPhase -ne "diagnostic" -and
    $BestCheckpointSeed -le $evaluationEnd -and
    $EvalSeed -le $selectionEnd
) {
    throw "Checkpoint-selection seeds $BestCheckpointSeed..$selectionEnd overlap evaluation seeds $EvalSeed..$evaluationEnd"
}
$checkpointRole = if ($SelectBestCheckpoint -eq 1) { "best" } else { "actor" }

$common = @(
    "--env_name", "IICMSMobility",
    "--scenario_name", "PureMobility",
    "--mobility", "rwp",
    "--ue_speed", "$Speed",
    "--slot_dt", "0.5",
    "--pathloss_model", "legacy",
    "--csi_aging", "0",
    "--obs_include_xy", "$ObsIncludeXy",
    "--obs_frames", "$ObsFrames",
    "--cluster_sizes", "2,3,4,5",
    "--cluster_action_mode", $ClusterActionMode,
    "--reward_mode", $RewardMode,
    "--objective_horizon", "$Episodes",
    "--reward_phi1", "0.5",
    "--reward_phi2", "1.0",
    "--rate_req_lo_mbps", "40",
    "--rate_req_hi_mbps", "40",
    "--min_ue_se_bps_hz", "$MinUeSeBpsHz",
    "--qos_penalty_weight", "$QosPenaltyWeight",
    "--backhaul_jt_model", "bottleneck",
    "--num_agents", "30",
    "--num_aps", "30",
    "--antennas_per_ap", "4",
    "--map_size", "1000",
    "--ap_min_distance", "50",
    "--backhaul_lo", "550000000",
    "--backhaul_hi", "650000000",
    "--fronthaul_queue", "$FronthaulQueue",
    "--fronthaul_buffer_s", "$FronthaulBufferS",
    "--episode_length", "$Episodes",
    "--num_env_steps", "$Steps",
    "--n_rollout_threads", "$Envs",
    "--vec_env", "dummy",
    "--n_training_threads", "4",
    "--ppo_epoch", "5",
    "--save_interval", "$SaveInterval",
    "--critic_hidden_size", "128",
    "--tf_n_head", "4",
    "--tf_n_layers", "2",
    "--tf_ff_mult", "4"
)
if ($SaveCheckpointSnapshots -eq 1) {
    $common += "--save_checkpoint_snapshots"
}

foreach ($seed in $Seeds) {
    if ($Algos -eq "both" -or $Algos -eq "mlp") {
        $mlp = $common + @(
            "--actor_backbone", "mlp",
            "--critic_backbone", $MlpCriticBackbone,
            "--hidden_size", "$MlpHidden",
            "--actor_temporal_context", $MlpActorTemporalContext,
            "--lr", "$MlpLr",
            "--critic_lr", "$MlpCriticLr",
            "--entropy_coef", "$MlpEntropyCoef",
            "--seed", "$seed",
            "--experiment_name", "${Tag}_mlp_s$seed"
        )
        if ($ArchitectureProfile -eq "native") {
            $mlp += @(
                "--disable_actor_token_norm",
                "--joint_action_head", "mlp_factorized"
            )
        } elseif ($ArchitectureProfile -eq "direct") {
            $mlp += @(
                "--disable_actor_token_norm",
                "--joint_action_head", "flat"
            )
        } else {
            $mlp += @("--joint_action_head", $JointActionHead)
        }
        if ($Cuda -eq 0) { $mlp += "--cuda" }
        Write-Host "===== seed $seed : MAPPO-MLP ====="
        & $Python train_main.py @mlp
        if ($LASTEXITCODE -ne 0) { throw "MLP training failed for seed $seed" }
        $mlpRunDir = "results/IICMSMobility/PureMobility/mappo/${Tag}_mlp_s${seed}/run1"
    }
    if ($Algos -eq "both" -or $Algos -eq "tf") {
        $tf = $common + @(
            "--actor_backbone", "transformer",
            "--critic_backbone", $TfCriticBackbone,
            "--hidden_size", "$TfHidden",
            "--tf_d_model", "$TfDModel",
            "--tf_pool", "$TfPool",
            "--actor_temporal_context", $TfActorTemporalContext,
            "--lr", "$TfLr",
            "--critic_lr", "$TfCriticLr",
            "--entropy_coef", "$TfEntropyCoef",
            "--seed", "$seed",
            "--experiment_name", "${Tag}_tf_actor_s$seed"
        )
        if ($ArchitectureProfile -eq "native" -or
            $ArchitectureProfile -eq "direct") {
            $tf += @(
                "--joint_action_head", "relational_subset",
                "--tf_relation_dim", "$TfRelationDim"
            )
        } else {
            $tf += @("--joint_action_head", $JointActionHead)
        }
        if ($TfUseCls -eq 1) { $tf += "--tf_use_cls" }
        if ($Cuda -eq 0) { $tf += "--cuda" }
        Write-Host "===== seed $seed : MAPPO-Transformer Actor ====="
        & $Python train_main.py @tf
        if ($LASTEXITCODE -ne 0) { throw "Transformer training failed for seed $seed" }
        $tfRunDir = "results/IICMSMobility/PureMobility/mappo/${Tag}_tf_actor_s${seed}/run1"
    }
    if ($Algos -in @("orig", "original", "all")) {
        $orig = $common + @(
            "--actor_backbone", "mlp",
            "--critic_backbone", "mlp",
            "--disable_actor_token_norm",
            "--hidden_size", "$PaperHidden",
            "--critic_hidden_size", "$PaperHidden",
            "--lr", "$PaperLr",
            "--critic_lr", "$PaperCriticLr",
            "--entropy_coef", "$PaperEntropyCoef",
            "--ppo_epoch", "$PaperPpoEpoch",
            "--joint_action_head", "paper_factorized",
            "--separate_policy",
            "--original_mappo",
            "--seed", "$seed",
            "--experiment_name", "${Tag}_orig_mappo_s$seed"
        )
        if ($Cuda -eq 0) { $orig += "--cuda" }
        Write-Host "===== seed $seed : Original-paper MAPPO ====="
        & $Python train_main.py @orig
        if ($LASTEXITCODE -ne 0) { throw "Original MAPPO training failed for seed $seed" }
        $origRunDir = "results/IICMSMobility/PureMobility/mappo/${Tag}_orig_mappo_s${seed}/run1"
    }
    if ($AutoEval -eq 1) {
        if ($Algos -eq "both" -or $Algos -eq "mlp") {
            $mlpRunDir = "results/IICMSMobility/PureMobility/mappo/${Tag}_mlp_s${seed}/run1"
        }
        if ($Algos -eq "both" -or $Algos -eq "tf") {
            $tfRunDir = "results/IICMSMobility/PureMobility/mappo/${Tag}_tf_actor_s${seed}/run1"
        }
        if ($Algos -in @("orig", "original", "all")) {
            $origRunDir = "results/IICMSMobility/PureMobility/mappo/${Tag}_orig_mappo_s${seed}/run1"
        }
        if ($SelectBestCheckpoint -eq 1) {
            if ($Algos -eq "both" -or $Algos -eq "mlp") {
                Write-Host "===== select best checkpoint: MLP seed $seed ====="
                & $Python select_best_checkpoint.py --run_dir $mlpRunDir --seed_start $BestCheckpointSeed --episodes $BestCheckpointEpisodes --horizon $EvalHorizon --metric $BestCheckpointMetric --max_qos_violation_rate $BestCheckpointMaxQos --qos_penalty_weight $QosPenaltyWeight
                if ($LASTEXITCODE -ne 0) { throw "MLP checkpoint selection failed for seed $seed" }
            }
            if ($Algos -eq "both" -or $Algos -eq "tf") {
                Write-Host "===== select best checkpoint: TF seed $seed ====="
                & $Python select_best_checkpoint.py --run_dir $tfRunDir --seed_start $BestCheckpointSeed --episodes $BestCheckpointEpisodes --horizon $EvalHorizon --metric $BestCheckpointMetric --max_qos_violation_rate $BestCheckpointMaxQos --qos_penalty_weight $QosPenaltyWeight
                if ($LASTEXITCODE -ne 0) { throw "Transformer checkpoint selection failed for seed $seed" }
            }
        }
        if ($Algos -eq "both" -or $Algos -eq "mlp") {
            Write-Host "===== evaluate seed $seed : MLP deterministic ====="
            & $Python evaluate_iicms.py --run_dir $mlpRunDir --seed_start $EvalSeed --episodes $EvalEpisodes --horizon $EvalHorizon --phase $EvalPhase --checkpoint $checkpointRole --out "$mlpRunDir/evaluation_deterministic"
            if ($LASTEXITCODE -ne 0) { throw "MLP deterministic evaluation failed for seed $seed" }
            Write-Host "===== evaluate seed $seed : MLP stochastic ====="
            & $Python evaluate_iicms.py --run_dir $mlpRunDir --seed_start $EvalSeed --episodes $EvalEpisodes --horizon $EvalHorizon --phase $EvalPhase --checkpoint $checkpointRole --stochastic --out "$mlpRunDir/evaluation_stochastic"
            if ($LASTEXITCODE -ne 0) { throw "MLP stochastic evaluation failed for seed $seed" }
            if ($EvalBaselines -eq 1) {
                Write-Host "===== evaluate seed $seed : MLP $BaselineLabel ====="
                & $Python evaluate_iicms.py --run_dir $mlpRunDir --seed_start $EvalSeed --episodes $EvalEpisodes --horizon $EvalHorizon --phase $EvalPhase --fixed_action $BaselineAction --policy_label $BaselineLabel --out "$mlpRunDir/baseline_${BaselineLabel}_deterministic"
                if ($LASTEXITCODE -ne 0) { throw "MLP baseline evaluation failed for seed $seed" }
            }
        }
        if ($Algos -eq "both" -or $Algos -eq "tf") {
            Write-Host "===== evaluate seed $seed : TF deterministic ====="
            & $Python evaluate_iicms.py --run_dir $tfRunDir --seed_start $EvalSeed --episodes $EvalEpisodes --horizon $EvalHorizon --phase $EvalPhase --checkpoint $checkpointRole --out "$tfRunDir/evaluation_deterministic"
            if ($LASTEXITCODE -ne 0) { throw "Transformer deterministic evaluation failed for seed $seed" }
            Write-Host "===== evaluate seed $seed : TF stochastic ====="
            & $Python evaluate_iicms.py --run_dir $tfRunDir --seed_start $EvalSeed --episodes $EvalEpisodes --horizon $EvalHorizon --phase $EvalPhase --checkpoint $checkpointRole --stochastic --out "$tfRunDir/evaluation_stochastic"
            if ($LASTEXITCODE -ne 0) { throw "Transformer stochastic evaluation failed for seed $seed" }
            if ($EvalBaselines -eq 1) {
                Write-Host "===== evaluate seed $seed : TF $BaselineLabel ====="
                & $Python evaluate_iicms.py --run_dir $tfRunDir --seed_start $EvalSeed --episodes $EvalEpisodes --horizon $EvalHorizon --phase $EvalPhase --fixed_action $BaselineAction --policy_label $BaselineLabel --out "$tfRunDir/baseline_${BaselineLabel}_deterministic"
                if ($LASTEXITCODE -ne 0) { throw "Transformer baseline evaluation failed for seed $seed" }
            }
        }
        if ($Algos -in @("orig", "original", "all")) {
            Write-Host "===== evaluate seed $seed : Original-paper MAPPO deterministic ====="
            & $Python evaluate_iicms.py --run_dir $origRunDir --seed_start $EvalSeed --episodes $EvalEpisodes --horizon $EvalHorizon --phase $EvalPhase --checkpoint actor --out "$origRunDir/evaluation_deterministic"
            if ($LASTEXITCODE -ne 0) { throw "Original MAPPO deterministic evaluation failed for seed $seed" }
            Write-Host "===== evaluate seed $seed : Original-paper MAPPO stochastic ====="
            & $Python evaluate_iicms.py --run_dir $origRunDir --seed_start $EvalSeed --episodes $EvalEpisodes --horizon $EvalHorizon --phase $EvalPhase --checkpoint actor --stochastic --out "$origRunDir/evaluation_stochastic"
            if ($LASTEXITCODE -ne 0) { throw "Original MAPPO stochastic evaluation failed for seed $seed" }
            if ($EvalBaselines -eq 1) {
                Write-Host "===== evaluate seed $seed : Original MAPPO $BaselineLabel ====="
                & $Python evaluate_iicms.py --run_dir $origRunDir --seed_start $EvalSeed --episodes $EvalEpisodes --horizon $EvalHorizon --phase $EvalPhase --fixed_action $BaselineAction --policy_label $BaselineLabel --out "$origRunDir/baseline_${BaselineLabel}_deterministic"
                if ($LASTEXITCODE -ne 0) { throw "Original MAPPO baseline evaluation failed for seed $seed" }
            }
        }
    }
}

if ($PlotResults -eq 1) {
    if (-not $FigureDir) {
        $FigureDir = "results/IICMSMobility/PureMobility/mappo/${Tag}_figures"
    }
    New-Item -ItemType Directory -Force -Path $FigureDir | Out-Null
    $plotFamilies = @()
    if ($Algos -eq "both" -or $Algos -eq "mlp") {
        $plotFamilies += "mlp"
    }
    if ($Algos -eq "both" -or $Algos -eq "tf") {
        $plotFamilies += "tf_actor"
    }
    if ($Algos -in @("orig", "original", "all")) {
        $plotFamilies += "orig_mappo"
    }
    if ($plotFamilies.Count -gt 0) {
        $seedCsv = ($Seeds -join ",")
        $familyCsv = ($plotFamilies -join ",")
        Write-Host "===== plotting training curves ====="
        & $Python plot_training_curves.py --tag $Tag --seeds $seedCsv --families $familyCsv --metric $PlotMetric --run latest --out "$FigureDir/${Tag}_training_curves.png"
        if ($LASTEXITCODE -ne 0) {
            Write-Warning "Training-curve plotting failed"
        }
        if ($AutoEval -eq 1) {
            Write-Host "===== plotting per-UE SE CDF ====="
            & $Python plot_seed_se_cdf.py --tag $Tag --seeds $seedCsv --families $familyCsv --metric $PlotMetric --run latest --out "$FigureDir/${Tag}_${PlotMetric}_cdf.png"
            if ($LASTEXITCODE -ne 0) {
                Write-Warning "CDF plotting failed"
            }
        }
    }
}

$packageArgs = @("package_results.py", "--tag", $Tag, "--log-dir", "logs_iicms", "--mode", $PackageMode)
foreach ($includeTag in $PackageIncludeTags) {
    if ($includeTag) { $packageArgs += @("--include-tag", $includeTag) }
}
& $Python @packageArgs
