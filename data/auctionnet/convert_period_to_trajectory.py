#!/usr/bin/env python3
"""Convert AuctionNet period CSVs to sembid trajectory CSVs.

This wraps
``external/auctionnet_repo/strategy_train_env/bidding_train_env/train_data_generator/train_data_generator.py::TrainDataGenerator``
so the project CLI has a single command for
``data/<level>/raw/period-N.csv`` → ``data/<level>/train/trajectory_data.csv``.

The original TrainDataGenerator is parameterless — it points at
``./data/traffic`` and writes outputs to ``./data/traffic/training_data_rlData_folder``.
We re-use its pure ``_generate_train_data`` method directly and write the
output where the project layout expects it.

Usage::

    PYTHONPATH=src \
    python scripts/data/convert_period_to_trajectory.py \
        --level mid \
        --raw-dir data/mid/raw \
        --output data/mid/train/trajectory_data.csv
"""

from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

# Make project + auctionnet_repo imports work.
_REPO_ROOT = Path(__file__).resolve().parents[2]
# ``bidding_train_env`` is a top-level package under
# ``external/auctionnet_repo/strategy_train_env``, not under
# ``external/auctionnet_repo`` directly.
_AUCTIONNET_STRATEGY = _REPO_ROOT / "external" / "auctionnet_repo" / "strategy_train_env"
for p in (str(_REPO_ROOT), str(_AUCTIONNET_STRATEGY)):
    if p not in sys.path:
        sys.path.insert(0, p)

import pandas as pd  # noqa: E402

from bidding_train_env.train_data_generator.train_data_generator import (  # noqa: E402
    TrainDataGenerator,
)


def _read_period_csv(csv_path: Path) -> pd.DataFrame:
    """Read a 18-column AuctionNet period CSV (no surprises)."""
    return pd.read_csv(csv_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Convert AuctionNet period CSVs to sembid trajectory CSVs. "
            "Wraps TrainDataGenerator._generate_train_data and writes a "
            "single combined trajectory_data.csv."
        )
    )
    parser.add_argument(
        "--level",
        required=True,
        choices=["high", "mid", "low"],
        help="CPA level (informational, written to the manifest).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        required=True,
        help="Directory containing period-{N}.csv files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output trajectory CSV path (14-column RL schema).",
    )
    args = parser.parse_args()

    period_csvs = sorted(glob.glob(str(args.raw_dir / "period-*.csv")))
    if not period_csvs:
        raise SystemExit(f"No period-*.csv files in {args.raw_dir}")

    print(f"Found {len(period_csvs)} period CSV(s) in {args.raw_dir}")
    gen = TrainDataGenerator()
    frames = []
    for csv_path in period_csvs:
        print(f"  reading {csv_path} ...")
        df = _read_period_csv(Path(csv_path))
        print(f"    rows={len(df):,}, advertisers={df.advertiserNumber.nunique()}, "
              f"timesteps={df.timeStepIndex.nunique()}, "
              f"CPA=[{df.CPAConstraint.min()}, {df.CPAConstraint.max()}]")
        print(f"  generating trajectory (this can take a few minutes) ...")
        traj = gen._generate_train_data(df)
        print(f"    trajectory rows={len(traj):,}, cols={list(traj.columns)}")
        frames.append(traj)
        del df

    combined = pd.concat(frames, axis=0, ignore_index=True)
    print(
        f"combined trajectory: rows={len(combined):,}, "
        f"periods={combined.deliveryPeriodIndex.nunique()}, "
        f"advertisers={combined.advertiserNumber.nunique()}"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(args.output, index=False)
    print(f"wrote {args.output} ({args.output.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
