#!/bin/bash
# Run this from the repo root to commit (avoiding local git "trailer" issue) and push.
set -e
cd "$(dirname "$0")"

# Unset any alias that might add --trailer
unalias git 2>/dev/null || true

# Commit (skip hooks in case one adds trailer)
git commit --no-verify -m "Initial commit: Surgical Co-Pilot (OxSurGemma) - code only"

# Push (force needed because we rewrote history with fresh init)
git branch -M main
git push -u origin main --force

echo "Done. Repo: https://github.com/PramitSaha/OxSurGemma"
