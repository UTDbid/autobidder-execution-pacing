"""Content fingerprint of a generated profile's parquet outputs.

Hashes the DATA (column values), not the parquet file bytes - parquet embeds
non-deterministic metadata (timestamps, compression layout) that would make a
raw file hash spuriously differ across identical-data runs. Used as the
determinism gate (two runs -> identical fingerprint) and, later, the BN
identity gate (flags-off == baseline).

Usage:  python -m t9sim.fingerprint output/golden
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from .paths import OUTPUT_DIR

_MASK = (1 << 64) - 1


def fp_frame(df: pd.DataFrame) -> int:
    # order-sensitive content hash; generation order is deterministic (seed)
    cols = sorted(df.columns)
    return int(pd.util.hash_pandas_object(df[cols], index=False).sum() & _MASK)


def fingerprint(d) -> int:
    p = Path(d)
    if not p.is_absolute() and not p.exists():
        p = OUTPUT_DIR / d
    files = sorted(p.glob("*.parquet"))
    if not files:
        print(f"no parquet files in {p}")
        return 0
    total = 0
    for f in files:
        df = pq.read_table(f).to_pandas()
        h = fp_frame(df)
        total = (total + h) & _MASK
        print(f"  {f.name:<26} rows={len(df):>10,}  fp={h:#018x}")
    print(f"COMBINED fingerprint = {total:#018x}")
    return total


if __name__ == "__main__":
    fingerprint(sys.argv[1] if len(sys.argv) > 1 else "golden")
