#!/usr/bin/env python3
"""Convert official raw period CSVs to 14-col trajectory shards.

Mirrors ``TrainDataGenerator._generate_train_data`` from the official repo
(strategy_train_env/bidding_train_env/train_data_generator/
train_data_generator.py:41-153). Reads period CSVs ONE AT A TIME to avoid
loading the whole 20+ GB raw dataset into memory.

Outputs either a single concatenated ``trajectory_data.csv`` OR per-period
shards ``trajectory_data__period-N.csv`` plus a manifest. The per-period
shard form is recommended for 20+ GB raw inputs.

Usage::

    PYTHONPATH=external/auctionnet_official_<ts>:src \
    python scripts/data/auctionnet_official/convert_official_raw_to_trajectory.py \
        --raw-dir data/mid/raw_official_20gb \
        --output-dir data/mid/train_official_20gb \
        --manifest data/manifests/convert_mid_official_20gb.json \
        --shard-mode shards
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

# Path bootstrap (same convention as generate_official_raw.py).
_REPO_ROOT = Path(__file__).resolve().parents[3]
_OFFICIAL_PATH = Path(os.environ.get(
    "AUCTIONNET_OFFICIAL_PATH",
    str(_REPO_ROOT / "external" / "auctionnet_official_20260618_091733"),
))
if not _OFFICIAL_PATH.exists():
    raise SystemExit(f"Official repo not found: {_OFFICIAL_PATH}")
if str(_OFFICIAL_PATH) not in sys.path:
    sys.path.insert(0, str(_OFFICIAL_PATH))
# strategy_train_env/ contains bidding_train_env.* (TrainDataGenerator lives there).
_STRATEGY_TRAIN_ENV = _OFFICIAL_PATH / "strategy_train_env"
if _STRATEGY_TRAIN_ENV.exists() and str(_STRATEGY_TRAIN_ENV) not in sys.path:
    sys.path.insert(0, str(_STRATEGY_TRAIN_ENV))
if str(_REPO_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT / "src"))

import pandas as pd  # noqa: E402

# TrainDataGenerator (the OFFICIAL converter).
from bidding_train_env.train_data_generator.train_data_generator import (  # noqa: E402
    TrainDataGenerator,
)


def _convert_one(
    period_csv: Path,
    out_csv: Path,
    *,
    level_cpa_range: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Convert one period CSV to trajectory via official TrainDataGenerator."""
    df_raw = pd.read_csv(period_csv)
    gen = TrainDataGenerator()
    traj = gen._generate_train_data(df_raw)
    traj.to_csv(out_csv, index=False)
    return {
        "period_csv": str(period_csv),
        "trajectory_csv": str(out_csv),
        "raw_rows": len(df_raw),
        "trajectory_rows": len(traj),
        "raw_size_bytes": period_csv.stat().st_size,
        "trajectory_size_bytes": out_csv.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="convert_official_raw_to_trajectory",
        description=(
            "Convert official AuctionNet raw period CSVs (18 cols) into 14-col "
            "trajectory shards via TrainDataGenerator._generate_train_data."
        ),
    )
    parser.add_argument("--raw-dir", type=Path, required=True,
                        help="Directory containing period-<N>.csv raw files")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="Where to write trajectory_data__period-<N>.csv shards")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--shard-mode", choices=["shards", "concat"],
                        default="shards",
                        help="shards: one trajectory CSV per period; concat: merge all")
    args = parser.parse_args()

    raw_dir: Path = args.raw_dir.expanduser().resolve()
    output_dir: Path = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path: Path = args.manifest.expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)

    period_files = sorted(raw_dir.glob("period-*.csv"))
    if not period_files:
        raise SystemExit(f"No period-*.csv found under {raw_dir}")

    started = time.time()
    results = []
    total_raw_bytes = 0
    total_traj_bytes = 0
    total_traj_rows = 0
    concat_frames = []

    for pf in period_files:
        out = output_dir / f"trajectory_data__{pf.stem}.csv"
        info = _convert_one(pf, out)
        results.append(info)
        total_raw_bytes += info["raw_size_bytes"]
        total_traj_bytes += info["trajectory_size_bytes"]
        total_traj_rows += info["trajectory_rows"]
        if args.shard_mode == "concat":
            concat_frames.append(pd.read_csv(out))
        print(f"[convert] {pf.name} -> {out.name} "
              f"({info['raw_rows']:,} raw -> {info['trajectory_rows']:,} traj)")

    concat_path = None
    if args.shard_mode == "concat" and concat_frames:
        concat_path = output_dir / "trajectory_data.csv"
        full = pd.concat(concat_frames, ignore_index=True)
        full.to_csv(concat_path, index=False)
        print(f"[concat] wrote {concat_path} ({len(full):,} rows, "
              f"{concat_path.stat().st_size / 1024**3:.2f} GB)")

    manifest = {
        "task": "convert_official_raw_to_trajectory",
        "raw_dir": str(raw_dir),
        "output_dir": str(output_dir),
        "shard_mode": args.shard_mode,
        "concat_path": str(concat_path) if concat_path else None,
        "num_periods": len(period_files),
        "total_raw_bytes": total_raw_bytes,
        "total_trajectory_bytes": total_traj_bytes,
        "total_trajectory_rows": total_traj_rows,
        "schema": [
            "deliveryPeriodIndex", "advertiserNumber", "advertiserCategoryIndex",
            "budget", "CPAConstraint", "realAllCost", "realAllConversion",
            "timeStepIndex", "state", "action", "reward", "reward_continuous",
            "done", "next_state",
        ],
        "per_period": results,
        "elapsed_s": round(time.time() - started, 1),
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"MANIFEST: {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())