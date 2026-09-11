$ErrorActionPreference = "Stop"
$Root = Resolve-Path (Join-Path $PSScriptRoot "..")
$Venv = Join-Path $Root ".venv"
$Python = Join-Path $Venv "Scripts\python.exe"

if (-not (Test-Path $Python)) {
    Write-Host "Creating virtual environment..."
    python -m venv $Venv
}

Write-Host "Installing EcoSort dependencies..."
& $Python -m pip install --upgrade pip
& $Python -m pip install -r (Join-Path $Root "requirements.txt")

Write-Host "Setup complete."
