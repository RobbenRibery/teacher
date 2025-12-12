#!/bin/bash
# Test script for student rollout

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

python scripts/test_student_rollout.py "$@"

