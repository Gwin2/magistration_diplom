$ErrorActionPreference = "Stop"

if (-not (Test-Path ".git")) {
  git init
}

git config core.autocrlf true
git config pull.rebase false

python -m pip install pre-commit
pre-commit install

Write-Host "Git project prepared: hooks installed."
