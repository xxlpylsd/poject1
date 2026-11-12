param(
    [int]$Seed = 1,
    [int]$Steps = 600000,
    [ValidateSet("both", "mlp", "tf")]
    [string]$Algos = "both",
    [int]$Envs = 4,
    [string]$Python = "python",
    [switch]$Cpu,
    [switch]$Smoke
)

$ErrorActionPreference = "Stop"
$comparison = Join-Path $PSScriptRoot "run_iicms_comparison.ps1"
if ($Smoke) {
    if (-not $PSBoundParameters.ContainsKey("Steps")) { $Steps = 800 }
    if (-not $PSBoundParameters.ContainsKey("Envs")) { $Envs = 1 }
    if (-not $PSBoundParameters.ContainsKey("Cpu")) { $Cpu = $true }
}
$cudaMode = if ($Cpu) { 0 } else { 1 }
$validationMode = if ($Smoke) { 0 } else { 1 }
$snapshotMode = if ($Smoke) { 0 } else { 1 }
$selectionMode = if ($Smoke) { 0 } else { 1 }
$evaluationMode = if ($Smoke) { 0 } else { 1 }
$plotMode = if ($Smoke) { 0 } else { 1 }

$arguments = @{
    Seeds = @($Seed)
    Steps = $Steps
    Episodes = 100
    Envs = $Envs
    Cuda = $cudaMode
    Python = $Python
    Algos = $Algos
    RewardMode = "team_rate"
    QosPenaltyWeight = 0.0
    MinUeSeBpsHz = 0.0
    Tag = "rate_contention_neighbor3_600000"
    ArchitectureProfile = "main"
    JointActionHead = "subset_conditional"
    ClusterActionMode = "subset"
    ObsIncludeXy = 1
    ObsFrames = 4
    ObsNeighborUes = 3
    ObsApCompetition = 1
    Speed = 15.0
    FronthaulQueue = 1
    FronthaulBufferS = 0.5
    MlpHidden = 192
    MlpLr = 2e-4
    MlpCriticLr = 1e-4
    MlpEntropyCoef = 0.005
    MlpCriticBackbone = "mlp"
    TfHidden = 128
    TfDModel = 64
    TfLr = 1e-4
    TfCriticLr = 1e-4
    TfEntropyCoef = 0.005
    TfCriticBackbone = "mlp"
    TfPool = "mean"
    TfUseCls = 0
    TrainingValidation = $validationMode
    ValidationIntervalSteps = 25000
    ValidationEpisodes = 20
    ValidationSeed = 91000
    ValidationHorizon = 100
    ValidationWindow = 5
    ValidationPlateauRelativeGain = 0.005
    SaveInterval = 10
    SaveCheckpointSnapshots = $snapshotMode
    SelectBestCheckpoint = $selectionMode
    BestCheckpointMetric = "system_se"
    BestCheckpointEpisodes = 20
    BestCheckpointSeed = 91000
    AutoEval = $evaluationMode
    EvalPhase = "test"
    EvalSeed = 93000
    EvalEpisodes = 20
    EvalHorizon = 100
    EvalBaselines = $evaluationMode
    PlotResults = $plotMode
    PlotMetric = "rate"
}

& $comparison @arguments
if ($LASTEXITCODE -ne 0) {
    throw "Contention experiment failed with exit code $LASTEXITCODE"
}
