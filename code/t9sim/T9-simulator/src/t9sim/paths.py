"""The ONE filesystem anchor for the package.

config/, calibration/ and output/ live under the package root (one level above
src/), preserving the v6 co-location contract in a single place instead of the
old per-module ``Path(__file__).resolve().parents[1]`` magic offset.
"""
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parents[2]   # .../t9_sim
CONFIG_DIR = PKG_ROOT / "config"
CALIBRATION_DIR = PKG_ROOT / "calibration"
OUTPUT_DIR = PKG_ROOT / "output"
