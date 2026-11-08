param(
    [string]$Python = "python",
    [string]$Tag = "update"
)

$ErrorActionPreference = "Stop"
& $Python package_project.py --run-tests --tag $Tag
if ($LASTEXITCODE -ne 0) {
    throw "Project packaging failed"
}
