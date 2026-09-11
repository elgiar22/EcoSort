param(
    [switch]$NoPublish,
    [int]$Camera = 0,
    [string]$Source = "station-01"
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$Runner = Join-Path $Root "apps\local_camera.py"
$Model = Join-Path $Root "best_model.keras"

if (-not (Test-Path $Python)) {
    throw "Missing .venv. Run .\scripts\setup.ps1 first."
}
if (-not (Test-Path $Model)) {
    throw "Missing best_model.keras in the project root."
}

$env:KERAS_BACKEND = "torch"
$argsList = @($Runner, "--model", $Model, "--camera", $Camera, "--source", $Source)
if ($NoPublish) { $argsList += "--no-publish" }

& $Python @argsList
