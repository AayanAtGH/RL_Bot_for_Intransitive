"""Launcher script for the Intransitive Tkinter GUI."""

import os
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent
os.chdir(ROOT_DIR)
sys.path.insert(0, str(ROOT_DIR))

from intransitive.cli.gui import main

if __name__ == "__main__":
    main()
