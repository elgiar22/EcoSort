param(
    [int]$Port = 8501
)

$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Python = Join-Path $Root ".venv\Scripts\python.exe"
$App = Join-Path $Root "apps\streamlit_app.py"
$Model = Join-Path $Root "best_model.keras"

if (-not (Test-Path $Python)) {
    throw "Missing .venv. Run .\scripts\setup.ps1 first."
}
if (-not (Test-Path $Model)) {
    throw "Missing best_model.keras in the project root."
}

$env:KERAS_BACKEND = "torch"
& $Python -m streamlit run $App `
    --server.port $Port `
    --server.address localhost `
    --browser.gatherUsageStats false
