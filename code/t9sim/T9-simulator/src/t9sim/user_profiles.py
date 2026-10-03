"""Stage 1 - user pool (v6).

One row per user: A1-A5 surface features + LU1-LU6 latent parameters.
Latents are drawn hierarchically: LU1 archetype ~ Categorical(pi), then
LU2-LU4 ~ Beta, LU5 ~ LogNormal (unit reference median), LU6 ~ Dirichlet,
all with per-archetype parameters from archetypes.yaml.

A1 region / A2 city come from the iPinYou deduped-user shapes (IP);
A3-A5 from the device config (StatCounter targets, placeholders pending).

Generation is chunked so the 22M-user scale100m profile streams within 16 GB.

Independent draws (pure v6 baseline). The BN dependency layer is added later
edge-by-edge behind per-edge flags (BN_Build_Plan.md); no BN wiring here.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config_loader import split_shares

CHUNK = 1_000_000


def _user_chunk(cfg, start_id, n, rng, shapes):
    A = cfg.archetypes
    dev = cfg.benchmarks["device"]
    cats = cfg.categories
    arch_labels, arch_p = split_shares(A["shares"])

    out = {"user_id": np.arange(start_id, start_id + n, dtype="int64")}

    # ── A1 region, A2 city | region (iPinYou shapes) ────────────────
    regions, region_p, city_map = shapes
    a1 = rng.choice(regions, size=n, p=region_p)
    a2 = np.empty(n, dtype="int64")
    for region in np.unique(a1):
        mask = a1 == region
        cities, cp = city_map[region]
        a2[mask] = rng.choice(cities, size=int(mask.sum()), p=cp)
    out["region"], out["city"] = a1.astype("int64"), a2

    # ── A3 os, A4 os_version|os, A5 device_type, LU1 archetype ─────
    os_labels, os_p = split_shares(dev["os_split"])
    if cfg.bn_edges.get("os_spend"):            # BN edge #3: realistic iOS share
        ios = float(cfg.benchmarks["bn"]["os_split_ios"])
        os_labels, os_p = ["iOS", "Android"], np.array([ios, 1.0 - ios])
    d_labels, d_p = split_shares(dev["device_split"])
    a3 = rng.choice(os_labels, size=n, p=os_p)
    a5 = rng.choice(d_labels, size=n, p=d_p)
    lu1 = rng.choice(len(arch_labels), size=n, p=arch_p)
    a4 = np.empty(n, dtype=object)                  # A4 os_version | os (nested)
    for osname in os_labels:
        mask = a3 == osname
        v_labels, v_p = split_shares(dev["os_versions"][osname])
        a4[mask] = rng.choice(v_labels, size=int(mask.sum()), p=v_p)
    out["os"], out["os_version"], out["device_type"] = a3, a4, a5
    out["lu1_archetype"] = np.array(arch_labels, dtype=object)[lu1]

    # ── LU2-LU4 Beta, LU5 LogNormal, LU6 Dirichlet (per archetype) ──
    lu2 = np.empty(n)
    lu3 = np.empty(n)
    lu4 = np.empty(n)
    lu5 = np.empty(n)
    lu6 = np.empty((n, len(cats)))
    k = A["interest"]["concentration_k"]
    for i, name in enumerate(arch_labels):
        mask = lu1 == i
        m = int(mask.sum())
        if m == 0:
            continue
        prop = A["propensity"][name]
        lu2[mask] = rng.beta(*prop["click"], size=m)
        lu3[mask] = rng.beta(*prop["install"], size=m)
        lu4[mask] = rng.beta(*prop["payer"], size=m)
        ltv = A["ltv_multiplier"][name]
        if "fixed" in ltv:
            lu5[mask] = float(ltv["fixed"])
            if float(ltv["fixed"]) == 0.0:
                # structural non-spender => structural non-payer, preserving
                # the master invariant is_payer = 1 <=> ltv_value > 0
                lu4[mask] = 0.0
        else:
            lu5[mask] = rng.lognormal(ltv["mu"], ltv["sigma"], size=m)
        alpha = k * np.asarray(A["interest"]["centroids"][name])
        lu6[mask] = rng.dirichlet(alpha, size=m)
    out["lu2_click_prop"] = lu2
    out["lu3_install_prop"] = lu3
    out["lu4_payer_prob"] = lu4
    out["lu5_ltv_mult"] = lu5
    for j, c in enumerate(cats):
        out[f"lu6_{c}"] = lu6[:, j]

    return pd.DataFrame(out)


def generate_users(cfg, out_path=None):
    """Generate the user pool. Returns a DataFrame (small profiles) or
    streams chunks to parquet when out_path is given (full scale)."""
    n_users = int(cfg.profile["n_users"])
    rng = cfg.rng("users")
    regions, region_p = cfg.pmf("ipinyou_region_distribution.csv")
    shapes = (regions.astype("int64"), region_p, cfg.city_by_region())

    frames = []
    writer = None
    for start in range(0, n_users, CHUNK):
        n = min(CHUNK, n_users - start)
        df = _user_chunk(cfg, start, n, rng, shapes)
        if out_path is None:
            frames.append(df)
        else:
            import pyarrow as pa
            import pyarrow.parquet as pq
            table = pa.Table.from_pandas(df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_path, table.schema)
            writer.write_table(table)
    if writer is not None:
        writer.close()
        return None
    return pd.concat(frames, ignore_index=True)
