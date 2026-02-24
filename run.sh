#!/usr/bin/env bash
# Run Surgical Co-Pilot from the repository root.
# Usage: ./run.sh [--gradio] [--port 8585] [--medgemma 4b|27b] ...

set -e
cd "$(dirname "$0")"
export PYTHONPATH="${PYTHONPATH:+$PYTHONPATH:}$(pwd)"
exec python -m surgical_copilot.main "$@"
