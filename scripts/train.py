"""Training script that calls train_grpo.py main function."""
from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path to import train_grpo
sys.path.insert(0, str(Path(__file__).parent.parent))

from teacher_torch.train_grpo import main

if __name__ == "__main__":
    main()

