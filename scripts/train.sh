#!/bin/bash
# Training script for GRPO

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

python scripts/train.py "$@"

