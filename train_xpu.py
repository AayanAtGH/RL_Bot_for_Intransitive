"""Entry-point script to run AlphaZero training with Intel XPU / IPEX auto-detection."""

import os
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR))

from intransitive.rl.trainer import main

if __name__ == "__main__":
    main()
