#!/usr/bin/env python3
"""Collect reproducible raw-data scale statistics for the final evidence bundle."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import zipfile
from pathlib import Path

import pandas as pd


def count_lines_stream(handle, chunk_size: int = 16 * 1024 * 1024) -> int:
    count = 0
    while True:
        block = handle.read(chunk_size)
        if not block:
            return count
        count += block.count(b"\n")


def csv_file_stats(path: Path) -> dict:
    with path.open("rb") as handle:
        lines = count_lines_stream(handle)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        first_rows = [next(reader) for _ in range(48)]
    advertisers = {int(float(row["advertiserNumber"])) for row in first_rows}
    categories = {int(float(row["advertiserCategoryIndex"])) for row in first_rows}
    budgets = [float(row["budget"]) for row in first_rows]
    cpas = [float(row["CPAConstraint"]) for row in first_rows]
    data_rows = lines - 1
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "data_rows": data_rows,
        "advertisers": len(advertisers),
        "categories": len(categories),
        "unique_pvs": data_rows // len(advertisers),
        "time_steps": 48,
        "budget_total": sum(budgets),
        "budget_min": min(budgets),
        "budget_median": float(pd.Series(budgets).median()),
        "budget_max": max(budgets),
        "cpa_min": min(cpas),
        "cpa_median": float(pd.Series(cpas).median()),
        "cpa_max": max(cpas),
    }


def zip_csv_stats(archive: Path, member: str) -> dict:
    with zipfile.ZipFile(archive) as zf:
        info = zf.getinfo(member)
        with zf.open(member) as handle:
            lines = count_lines_stream(handle)
        with zf.open(member) as raw:
            text = io.TextIOWrapper(raw, encoding="utf-8", newline="")
            reader = csv.DictReader(text)
            first_rows = [next(reader) for _ in range(48)]
    advertisers = {int(float(row["advertiserNumber"])) for row in first_rows}
    categories = {int(float(row["advertiserCategoryIndex"])) for row in first_rows}
    budgets = [float(row["budget"]) for row in first_rows]
    cpas = [float(row["CPAConstraint"]) for row in first_rows]
    data_rows = lines - 1
    return {
        "archive": str(archive),
        "archive_bytes": archive.stat().st_size,
        "member": member,
        "member_uncompressed_bytes": info.file_size,
        "data_rows": data_rows,
        "advertisers": len(advertisers),
        "categories": len(categories),
        "unique_pvs": data_rows // len(advertisers),
        "time_steps": 48,
        "budget_total": sum(budgets),
        "budget_min": min(budgets),
        "budget_median": float(pd.Series(budgets).median()),
        "budget_max": max(budgets),
        "cpa_min": min(cpas),
        "cpa_median": float(pd.Series(cpas).median()),
        "cpa_max": max(cpas),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--legacy-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    legacy = args.legacy_root
    external = args.external_root
    project = args.project_root

    micro_paths = {
        "high_p7": legacy / "sembid_new/data/high/test/period-7.csv",
        "high_p8": legacy / "sembid_new/data/high/test/period-8.csv",
        "low_p7": legacy / "sembid_new/data/low/test/period-7.csv",
        "low_p8": legacy / "sembid_new/data/low/test/period-8.csv",
    }
    macro_archive_1112 = external / "datasets/AuctionNet_PlatformBid_periods_9_13_official/archives/autoBidding_aigb_track_data_period_11-12.zip"
    macro_archive_13 = external / "datasets/AuctionNet_PlatformBid_periods_9_13_official/archives/autoBidding_aigb_track_data_period_13.zip"

    ipy_audit = pd.read_csv(project / "00_admin/audits/ipinyou_clean/ipinyou_clean_campaign_audit.csv")
    ipy_manifest = pd.read_csv(project / "08_results/ipinyou/formal_v2/ipinyou_run_manifest.csv")
    t9_download = json.loads((project / "00_admin/audits/T9SIM_DOWNLOAD_AUDIT.json").read_text())
    t9_manifest = pd.read_csv(project / "08_results/t9sim/formal_10m/t9sim_formal_manifest.csv")
    t9_campaign = pd.read_csv(
        project / "08_results/t9sim/formal_10m/t9sim_formal_campaign_raw.csv",
        usecols=["seed", "condition", "campaign_id", "budget_multiplier", "reference", "kappa", "budget"],
    )
    t9_budget_base = t9_campaign[
        (t9_campaign["condition"] == "C1")
        & (t9_campaign["reference"] == "traffic_aware")
        & (t9_campaign["kappa"] == 0)
        & (t9_campaign["budget_multiplier"] == 1)
    ].drop_duplicates(["seed", "campaign_id"])

    ten_m_files = [item for item in t9_download["files"] if "_10m_seed" in item["file"]]
    raw = {
        "auctionnet_micro": {name: csv_file_stats(path) for name, path in micro_paths.items()},
        "auctionnet_macro": {
            "period11": zip_csv_stats(macro_archive_1112, "period-11.csv"),
            "period12": zip_csv_stats(macro_archive_1112, "period-12.csv"),
            "period13": zip_csv_stats(macro_archive_13, "period-13.csv"),
        },
        "ipinyou": {
            "campaigns": int(ipy_audit["campaign"].nunique()),
            "campaign_ids": [int(x) for x in sorted(ipy_audit["campaign"].unique())],
            "train_rows": int(ipy_audit["train_rows"].sum()),
            "pctr_training_rows": int(ipy_manifest["pctr_train_rows"].sum()),
            "calibration_rows": int(ipy_manifest["calibration_rows"].sum()),
            "test_rows": int(ipy_manifest["test_rows"].sum()),
            "train_clicks": int(ipy_audit["train_clicks"].sum()),
            "test_clicks": int(ipy_audit["test_clicks"].sum()),
            "train_conversions": int(ipy_audit["train_conversions"].sum()),
            "test_conversions": int(ipy_audit["test_conversions"].sum()),
            "train_rows_min": int(ipy_audit["train_rows"].min()),
            "train_rows_median": float(ipy_audit["train_rows"].median()),
            "train_rows_max": int(ipy_audit["train_rows"].max()),
            "test_rows_min": int(ipy_audit["test_rows"].min()),
            "test_rows_median": float(ipy_audit["test_rows"].median()),
            "test_rows_max": int(ipy_audit["test_rows"].max()),
            "raw_directory_bytes": sum(p.stat().st_size for p in (project / "02_data_raw/ipinyou").rglob("*") if p.is_file()),
            "clean_directory_bytes": sum(p.stat().st_size for p in (project / "03_data_clean/ipinyou").rglob("*") if p.is_file()),
        },
        "t9sim": {
            "generator_seeds": int(t9_manifest["seed"].nunique()),
            "conditions": sorted(t9_manifest["condition"].unique().tolist()),
            "underlying_rows_per_seed": 10_000_000,
            "underlying_rows_total": 10_000_000 * int(t9_manifest["seed"].nunique()),
            "view_rows_total": 10_000_000 * len(t9_manifest),
            "columns": 55,
            "campaigns_per_seed": int(t9_budget_base.groupby("seed")["campaign_id"].nunique().median()),
            "campaign_seed_units": int(t9_budget_base[["seed", "campaign_id"]].drop_duplicates().shape[0]),
            "training_rows_per_seed": int(t9_manifest.groupby("seed")["training_rows"].first().median()),
            "calibration_rows_per_seed": int(t9_manifest.groupby("seed")["calibration_rows"].first().median()),
            "evaluation_rows_per_seed": int(t9_manifest.groupby("seed")["evaluation_rows"].first().median()),
            "budget_per_campaign_min": float(t9_budget_base["budget"].min()),
            "budget_per_campaign_median": float(t9_budget_base["budget"].median()),
            "budget_per_campaign_max": float(t9_budget_base["budget"].max()),
            "budget_total_mean_per_seed": float(t9_budget_base.groupby("seed")["budget"].sum().mean()),
            "ten_m_archives_bytes": int(sum(item["size"] for item in ten_m_files)),
            "raw_directory_bytes": sum(p.stat().st_size for p in (project / "02_data_raw/t9sim").rglob("*") if p.is_file()),
            "clean_directory_bytes": sum(p.stat().st_size for p in (project / "03_data_clean/t9sim").rglob("*") if p.is_file()),
        },
    }

    # Unique AuctionNet scale: High and Low share each period's auction skeleton.
    micro_unique_pvs = raw["auctionnet_micro"]["high_p7"]["unique_pvs"] + raw["auctionnet_micro"]["high_p8"]["unique_pvs"]
    macro_unique_pvs = sum(item["unique_pvs"] for item in raw["auctionnet_macro"].values())
    raw["auctionnet_totals"] = {
        "unique_period_pvs": micro_unique_pvs + macro_unique_pvs,
        "unique_advertiser_opportunity_pairs": (micro_unique_pvs + macro_unique_pvs) * 48,
        "stored_micro_rows_high_plus_low": sum(item["data_rows"] for item in raw["auctionnet_micro"].values()),
        "stored_macro_rows": sum(item["data_rows"] for item in raw["auctionnet_macro"].values()),
        "note": "High and Low share the Period-7/8 auction skeleton and are not counted as independent opportunities.",
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(raw, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "top_level": list(raw)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
