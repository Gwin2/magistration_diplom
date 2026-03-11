#!/usr/bin/env bash
set -euo pipefail

if [ ! -d ".git" ]; then
  git init
fi

git config pull.rebase false

python -m pip install pre-commit
pre-commit install

echo "Git project prepared: hooks installed."
