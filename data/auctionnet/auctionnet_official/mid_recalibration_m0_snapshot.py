#!/usr/bin/env python3
"""Gate M0: snapshot the current canonical ``mid`` train/test files.

Non-destructive. For each file record size_bytes, mtime, sha256, schema
(column list), row_count, and a conversion/reward profile by reusing the
existing ``measure_conversion_profile`` functions. Writes one atomic JSON
manifest. Does not modify or delete any canonical file.

Usage::

    python scripts/data/auctionnet_official/mid_recalibration_m0_snapshot.py \\
        --run-id 20260715 \\
        --output data/manifests/mid_recalibration_m0_snapshot_20260715.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AOFF_DIR = _REPO_ROOT / "scripts" / "data" / "auctionnet_official"
if str(_AOFF_DIR) not in sys.path:
    sys.path.insert(0, str(_AOFF_DIR))

from measure_conversion_profile import measure_raw_csv, measure_trajectory_csv  # noqa: E402


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _schema(path: Path) -> list[str]:
    df = pd.read_csv(path, nrows=0)
    return list(df.columns)


def _snapshot_one(path: Path, kind: str, chunksize: int) -> dict[str, Any]:
    st = path.stat()
    print(f"[M0] snapshotting {kind}: {path} ({st.st_size} bytes)", flush=True)
    schema = _schema(path)
    print(f"[M0]   schema({len(schema)} cols): {schema}", flush=True)
    print(f"[M0]   computing sha256 ...", flush=True)
    digest = _sha256(path)
    print(f"[M0]   sha256={digest}", flush=True)
    print(f"[M0]   measuring profile (chunksize={chunksize}) ...", flush=True)
    if kind == "raw":
        profile = measure_raw_csv(path, chunksize)
    else:
        profile = measure_trajectory_csv(path, chunksize)
    row_count = int(profile.get("rows", 0))
    try:
        p_str = str(path.relative_to(_REPO_ROOT))
    except ValueError:
        p_str = str(path)
    return {
        "path": p_str,
        "kind": kind,
        "size_bytes": int(st.st_size),
        "mtime_iso": datetime.fromtimestamp(st.st_mtime, tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "sha256": digest,
        "schema": schema,
        "row_count": row_count,
        "profile": profile,
    }


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    with open(tmp, "rb") as fh:
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def main() -> int:
    p = argparse.ArgumentParser(description="M0 snapshot of canonical mid train/test")
    p.add_argument("--run-id", required=True)
    p.add_argument("--train-csv", type=Path,
                   default=_REPO_ROOT / "data" / "mid" / "train" / "trajectory_data.csv")
    p.add_argument("--test-csv", type=Path,
                   default=_REPO_ROOT / "data" / "mid" / "test" / "period-0.csv")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--chunksize", type=int, default=1_000_000)
    args = p.parse_args()

    for f in (args.train_csv, args.test_csv):
        if not f.exists():
            raise SystemExit(f"canonical file not found: {f}")

    manifest: dict[str, Any] = {
        "task": "mid_recalibration_m0_snapshot",
        "run_id": args.run_id,
        "recorded_at_iso": _now_iso(),
        "hostname": os.uname().nodename,
        "python": sys.executable,
        "train": _snapshot_one(args.train_csv.resolve(), "trajectory", args.chunksize),
        "test": _snapshot_one(args.test_csv.resolve(), "raw", args.chunksize),
        "note": "non-destructive snapshot; no canonical file was modified or deleted",
    }
    manifest["completed_at_iso"] = _now_iso()
    _write_json_atomic(args.output.resolve(), manifest)
    print(f"[M0] wrote manifest: {args.output}", flush=True)
    # Echo a compact summary for logs.
    print(json.dumps({
        "train_rows": manifest["train"]["row_count"],
        "train_sha256": manifest["train"]["sha256"],
        "test_rows": manifest["test"]["row_count"],
        "test_sha256": manifest["test"]["sha256"],
        "test_mean_pvalue": manifest["test"]["profile"].get("mean_pvalue"),
    }, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
