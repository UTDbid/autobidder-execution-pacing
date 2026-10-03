"""T9 v6 simulator orchestrator.

  pools (users / apps / campaigns)  ->  warm-up calibration (z-stats of
  log EV + k_global solved to the win-rate target)  ->  chunked auction
  generation (funnel on ALL rows; master uncensored)  ->  parquet +
  manifest + provenance table + validation report.

Condition views C1-C4 are produced on demand by censor.view(master, "Cx")
- they are masks over the master, not stored copies.

Usage:
  python -m t9sim.simulate [--profile golden|test|scale10m|scale50m|scale100m]
"""
from __future__ import annotations

import argparse
import json
import time

import pyarrow as pa
import pyarrow.parquet as pq

from . import user_profiles, validate
from .auctions import AuctionEngine
from .catalogue import build_apps, build_campaigns
from .config_loader import Config
from .paths import OUTPUT_DIR


def run(profile=None, gamma=None, out_name=None, seed=None, bn_edges=None, rho=None):
    t0 = time.time()
    cfg = Config(profile)
    if gamma is not None:                       # gamma-sweep override
        cfg.benchmarks["auction"]["gamma"] = float(gamma)
    if rho is not None:                         # reason-1c rho-sweep override
        cfg.benchmarks["auction"]["rho"] = float(rho)
    if seed is not None:                        # multi-seed CI override
        cfg.profile["seed"] = int(seed)
    if bn_edges:                                # BN per-edge flag override
        cfg.bn_edges = {**cfg.bn_edges, **bn_edges}
    if out_name is None:
        out = cfg.output_dir()
    else:
        out = OUTPUT_DIR / out_name
        out.mkdir(parents=True, exist_ok=True)
    n_auctions = int(cfg.profile["n_auctions"])
    chunk_size = int(cfg.profile["chunk_size"])
    print(f"profile={cfg.profile_name}: {n_auctions:,} auctions, "
          f"{cfg.profile['n_users']:,} users, {cfg.profile['n_apps']} apps, "
          f"{cfg.profile['n_campaigns']} campaigns (seed "
          f"{cfg.profile['seed']})")

    users = user_profiles.generate_users(cfg)
    apps = build_apps(cfg)
    campaigns = build_campaigns(cfg)
    users.to_parquet(out / "pool_users.parquet", index=False)
    apps.to_parquet(out / "pool_apps.parquet", index=False)
    campaigns.to_parquet(out / "pool_campaigns.parquet", index=False)
    print(f"pools written ({time.time() - t0:.1f}s)")

    engine = AuctionEngine(cfg, users, apps, campaigns)
    cal = engine.calibrate()
    print(f"warm-up: k_global={cal['k_global']:.4g}, "
          f"win_rate={cal['warmup_win_rate']:.3f} "
          f"(target {cfg.win_rate['target']})")

    writer = None
    done = 0
    chunk_idx = 0
    while done < n_auctions:
        n = min(chunk_size, n_auctions - done)
        rng = cfg.rng(f"auctions-{chunk_idx}")
        df = engine.chunk(n, rng)
        table = pa.Table.from_pandas(df, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(out / "auctions.parquet", table.schema)
        writer.write_table(table)
        done += n
        chunk_idx += 1
        print(f"  chunk {chunk_idx}: {done:,}/{n_auctions:,} "
              f"({time.time() - t0:.1f}s)")
    writer.close()

    try:
        cfg.provenance.to_csv(out / "provenance.csv", index=False)
    except PermissionError:           # file open in Excel etc.
        alt = out / "provenance_new.csv"
        cfg.provenance.to_csv(alt, index=False)
        print(f"WARNING: provenance.csv locked by another program - "
              f"wrote {alt.name} instead")
    # The manifest is what a downloaded dataset carries to identify itself, so it
    # must record the two settings that define the run's schema variant: rho (the
    # privateness dial) and the BN edge flags. Without them a dataset cannot be
    # told apart from one generated under a different configuration.
    manifest = {
        "profile": cfg.profile_name, **cfg.profile,
        "win_rate_target": cfg.win_rate["target"], **cal,
        "rho": float(cfg.benchmarks.get("auction", {}).get("rho", 0.0)),
        "bn_edges": {k: bool(v) for k, v in sorted(cfg.bn_edges.items())},
        "schema": "T9Sim Specification v10",
        "elapsed_s": round(time.time() - t0, 1),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2),
                                       encoding="utf-8")

    df_all = pq.read_table(out / "auctions.parquet").to_pandas()
    table, ok = validate.report(df_all, cfg)
    try:
        table.to_csv(out / "validation_report.csv", index=False)
    except PermissionError:
        table.to_csv(out / "validation_report_new.csv", index=False)
        print("WARNING: validation_report.csv locked - wrote "
              "validation_report_new.csv instead")
    print(f"\n{'ALL DIRECTION CHECKS PASS' if ok else 'DIRECTION CHECK FAILURES'}"
          f" - output in {out}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=None,
                    help="golden | test | scale10m | scale50m | scale100m "
                         "(default from profiles.yaml)")
    ap.add_argument("--bn-edge", action="append", default=[], metavar="NAME",
                    help="enable a BN edge (repeatable): pairing | os_spend | "
                         "payer_timing | hour | exposure")
    args = ap.parse_args()
    bn = {e: True for e in args.bn_edge}
    raise SystemExit(0 if run(args.profile, bn_edges=bn) else 1)


if __name__ == "__main__":
    main()
