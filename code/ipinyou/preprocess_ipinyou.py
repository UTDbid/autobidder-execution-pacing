#!/usr/bin/env python3
"""Deterministic Python-3 parser for the official iPinYou contest archive."""

from __future__ import annotations

import argparse
import bz2
from collections import Counter
from datetime import datetime
import json
from pathlib import Path
from typing import Iterable

import pyarrow as pa
import pyarrow.parquet as pq


RAW_SCHEMA = [
    "bidid", "timestamp", "logtype", "ipinyouid", "useragent", "ip",
    "region", "city", "adexchange", "domain", "url", "urlid", "slotid",
    "slotwidth", "slotheight", "slotvisibility", "slotformat", "slotprice",
    "creative", "bidprice", "payprice", "keypage", "advertiser", "usertag",
]
CAMPAIGNS = ("1458", "3358", "3386", "3427", "3476", "2259", "2261", "2821", "2997")
# The public benchmark convention (and the authors' make-ipinyou-data code) uses
# Seasons 2 and 3.  Season 1 has a different 22-column schema and is excluded.
CAMPAIGN_SEASONS = {
    "1458": "2nd", "3358": "2nd", "3386": "2nd", "3427": "2nd", "3476": "2nd",
    "2259": "3rd", "2261": "3rd", "2821": "3rd", "2997": "3rd",
}
OUT_SCHEMA = pa.schema([
    ("campaign_id", pa.string()), ("split", pa.string()), ("bidid", pa.string()),
    ("timestamp", pa.int64()), ("date", pa.string()), ("weekday", pa.int8()),
    ("hour", pa.int8()), ("os_browser", pa.string()), ("region", pa.string()),
    ("city", pa.string()), ("adexchange", pa.string()), ("domain", pa.string()),
    ("slotid", pa.string()), ("slotwidth", pa.int32()), ("slotheight", pa.int32()),
    ("slotvisibility", pa.string()), ("slotformat", pa.string()),
    ("slotprice", pa.float64()), ("creative", pa.string()),
    ("bidprice", pa.float64()), ("payprice", pa.float64()),
    ("keypage", pa.string()), ("usertag", pa.string()),
    ("click", pa.int8()), ("conversion", pa.int8()),
    ("source_file", pa.string()), ("source_line", pa.int64()),
])


def open_text(path: Path):
    return bz2.open(path, "rt", encoding="utf-8", errors="replace") if path.suffix == ".bz2" else path.open("r", encoding="utf-8", errors="replace")


def normalized_ua(value: str) -> str:
    text = value.lower()
    os_name = next((name for name in ("windows", "ios", "mac", "android", "linux") if name in text), "other")
    browser = next((name for name in ("chrome", "sogou", "maxthon", "safari", "firefox", "theworld", "opera", "ie") if name in text), "other")
    return f"{os_name}_{browser}"


def parse_time(value: str) -> tuple[int, str, int, int] | None:
    try:
        dt = datetime.strptime(value[:14], "%Y%m%d%H%M%S")
    except (ValueError, TypeError):
        return None
    return int(value[:14]), dt.strftime("%Y-%m-%d"), (dt.weekday() + 1) % 7, dt.hour


def season_files(root: Path, campaign: str, split: str, prefix: str | None = None) -> list[Path]:
    season = CAMPAIGN_SEASONS[campaign]
    if split == "train":
        assert prefix is not None
        pattern = f"training{season}/{prefix}.*.txt.bz2"
    elif split == "test":
        pattern = f"testing{season}/leaderboard.test.data.*.txt.bz2"
    else:
        raise ValueError(split)
    return sorted(path for path in root.glob(pattern) if path.is_file())


def event_keys(files: Iterable[Path], audit: Counter, label: str, campaign: str) -> set[str]:
    keys: set[str] = set()
    for path in files:
        with open_text(path) as handle:
            for line_number, line in enumerate(handle, 1):
                parts = line.rstrip("\n").split("\t")
                if len(parts) < len(RAW_SCHEMA):
                    audit[f"malformed_{label}_row"] += 1
                    continue
                if parts[22].strip() != campaign:
                    audit[f"other_campaign_{label}_row"] += 1
                    continue
                keys.add(parts[0] + "-" + parts[18])
                audit[f"{label}_event_rows"] += 1
    return keys


def clean_row(parts: list[str], *, campaign: str, split: str, source: Path, line_number: int, click: int, conversion: int, audit: Counter) -> dict | None:
    if len(parts) < len(RAW_SCHEMA):
        audit["malformed_column_count"] += 1
        return None
    time_parts = parse_time(parts[1])
    if time_parts is None:
        audit["malformed_timestamp"] += 1
        return None
    try:
        slotwidth, slotheight = int(parts[13] or 0), int(parts[14] or 0)
        slotprice = float(parts[17] or 0)
        bidprice = float(parts[19] or 0)
        payprice = float(parts[20] or 0)
    except ValueError:
        audit["malformed_numeric"] += 1
        return None
    if min(slotwidth, slotheight, slotprice, bidprice, payprice) < 0:
        audit["negative_numeric"] += 1
        return None
    advertiser = parts[22].strip()
    if advertiser and advertiser != campaign:
        audit["advertiser_mismatch"] += 1
        return None
    timestamp, date, weekday, hour = time_parts
    audit["accepted"] += 1
    audit["clicks"] += int(click)
    audit["conversions"] += int(conversion)
    return {
        "campaign_id": campaign, "split": split, "bidid": parts[0],
        "timestamp": timestamp, "date": date, "weekday": weekday, "hour": hour,
        "os_browser": normalized_ua(parts[4]), "region": parts[6] or "null",
        "city": parts[7] or "null", "adexchange": parts[8] or "null",
        "domain": parts[9] or "null", "slotid": parts[12] or "null",
        "slotwidth": slotwidth, "slotheight": slotheight,
        "slotvisibility": parts[15] or "null", "slotformat": parts[16] or "null",
        "slotprice": slotprice, "creative": parts[18] or "null",
        "bidprice": bidprice, "payprice": payprice,
        "keypage": parts[21] or "null", "usertag": parts[23].strip() or "null",
        "click": int(click), "conversion": int(conversion),
        "source_file": str(source), "source_line": line_number,
    }


def flush(writer: pq.ParquetWriter, rows: list[dict]) -> None:
    if rows:
        writer.write_table(pa.Table.from_pylist(rows, schema=OUT_SCHEMA))
        rows.clear()


def process_training(root: Path, campaign: str, output: Path) -> dict:
    audit = Counter()
    impression_files = season_files(root, campaign, "train", "imp")
    click_files = season_files(root, campaign, "train", "clk")
    conversion_files = season_files(root, campaign, "train", "conv")
    if not impression_files:
        raise FileNotFoundError(f"no training impressions for campaign {campaign}")
    keys = event_keys(click_files, audit, "click", campaign)
    conversion_keys = event_keys(conversion_files, audit, "conversion", campaign)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(output, OUT_SCHEMA, compression="zstd")
    rows: list[dict] = []
    try:
        for path in impression_files:
            with open_text(path) as handle:
                for line_number, line in enumerate(handle, 1):
                    audit["input_rows"] += 1
                    parts = line.rstrip("\n").split("\t")
                    event_key = parts[0] + "-" + parts[18] if len(parts) > 18 else ""
                    click = int(event_key in keys)
                    conversion = int(event_key in conversion_keys)
                    row = clean_row(parts, campaign=campaign, split="train", source=path,
                                    line_number=line_number, click=click, conversion=conversion, audit=audit)
                    if row is not None:
                        rows.append(row)
                    if len(rows) >= 100_000:
                        flush(writer, rows)
        flush(writer, rows)
    finally:
        writer.close()
    return {"campaign": campaign, "split": "train", "files": [str(path) for path in impression_files], "click_files": [str(path) for path in click_files], "conversion_files": [str(path) for path in conversion_files], "audit": dict(audit)}


def process_testing(root: Path, campaign: str, output: Path) -> dict:
    audit = Counter()
    files = season_files(root, campaign, "test")
    if not files:
        raise FileNotFoundError(f"no testing file for campaign {campaign}")
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(output, OUT_SCHEMA, compression="zstd")
    rows: list[dict] = []
    try:
        for path in files:
            with open_text(path) as handle:
                for line_number, line in enumerate(handle, 1):
                    audit["input_rows"] += 1
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < len(RAW_SCHEMA) + 2:
                        audit["malformed_test_column_count"] += 1
                        continue
                    try:
                        click = int(float(parts[-2]) > 0)
                        conversion = int(float(parts[-1]) > 0)
                    except ValueError:
                        audit["malformed_test_outcome"] += 1
                        continue
                    row = clean_row(parts[:len(RAW_SCHEMA)], campaign=campaign, split="test", source=path,
                                    line_number=line_number, click=click, conversion=conversion, audit=audit)
                    if row is not None:
                        rows.append(row)
                    if len(rows) >= 100_000:
                        flush(writer, rows)
        flush(writer, rows)
    finally:
        writer.close()
    return {"campaign": campaign, "split": "test", "files": [str(path) for path in files], "audit": dict(audit)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--campaign", choices=CAMPAIGNS, required=True)
    args = parser.parse_args()
    campaign_dir = args.output_root / args.campaign
    train = process_training(args.raw_root, args.campaign, campaign_dir / "train.parquet")
    test = process_testing(args.raw_root, args.campaign, campaign_dir / "test.parquet")
    report = {"schema_version": 1, "campaign": args.campaign, "train": train, "test": test}
    (campaign_dir / "preprocess_audit.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
