"""v6 config loading with provenance tracking.

Every calibratable leaf in the YAML configs is a wrapper map
``{value, route, source}`` (route: quoted | auto-calibrated | inferred).
``Config`` flattens wrappers to plain values and collects every
(param, value, route, source) row into a provenance registry - the raw
material for the FN-1 target-vs-achieved calibration table.

Path anchoring lives in paths.py (the single co-location anchor); this module
imports CONFIG_DIR / CALIBRATION_DIR / OUTPUT_DIR from there.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import yaml

from .paths import CALIBRATION_DIR, CONFIG_DIR, OUTPUT_DIR

_WRAPPER_KEYS = {"value", "route", "source"}


def _is_wrapper(node):
    return isinstance(node, dict) and "value" in node and \
        set(node) <= _WRAPPER_KEYS


def _flatten(node, path, registry):
    if _is_wrapper(node):
        registry.append({
            "param": path,
            "value": node["value"],
            "route": node.get("route", "untagged"),
            "source": node.get("source", ""),
        })
        return node["value"]
    if isinstance(node, dict):
        return {k: _flatten(v, f"{path}.{k}" if path else str(k), registry)
                for k, v in node.items()}
    return node


class Config:
    """Flattened config values + provenance registry + calibration shapes."""

    def __init__(self, profile=None):
        registry = []
        for name in ("benchmarks", "archetypes", "profiles", "validation"):
            raw = yaml.safe_load((CONFIG_DIR / f"{name}.yaml").read_text(
                encoding="utf-8"))
            setattr(self, name, _flatten(raw, name, registry))
        self.provenance = pd.DataFrame(registry)

        # merge solved calibration knobs (config/calibrated.yaml, written by
        # autocal.py) over the base values; provenance rows updated in place
        calib_path = CONFIG_DIR / "calibrated.yaml"
        if calib_path.exists():
            solved = yaml.safe_load(calib_path.read_text(
                encoding="utf-8"))["solved"]
            for path, info in solved.items():
                node = self
                keys = path.split(".")
                obj = getattr(node, keys[0])
                for k in keys[1:-1]:
                    obj = obj[k]
                obj[keys[-1]] = info["value"]
                m = self.provenance["param"] == path
                self.provenance.loc[m, "value"] = info["value"]
                self.provenance.loc[m, "source"] = info["note"]
                self.provenance.loc[m, "route"] = "auto-calibrated"

        prof_name = profile or self.profiles["default_profile"]
        if prof_name not in self.profiles["profiles"]:
            raise KeyError(f"unknown profile '{prof_name}'")
        self.profile_name = prof_name
        self.profile = self.profiles["profiles"][prof_name]
        self.win_rate = self.profiles["win_rate"]
        self.categories = self.archetypes["category_order"]

        # BN dependency layer (v7 funnel + v8 price): per-edge flags (default OFF)
        # + resolved CPTs. All-OFF reproduces the independent baseline exactly.
        _edge_defaults = {"pairing": False, "os_spend": False,
                          "payer_timing": False, "hour": False, "exposure": False,
                          # v8 price/competition (SSP) layer
                          "competition": False, "app_quality_price": False,
                          "exchange_price": False, "app_price": False,
                          "time_competition": False, "ev_market": False,
                          # reason-1c: hidden persistent rival-segment factor in LU7
                          "rival_hidden": False,
                          # pilot-2: hidden factor = group loading x user latent value
                          "rival_user_corr": False,
                          # v10: Private-Rival Market (K archetypes -> LU7 = max_k b_k)
                          "rival_pool": False,
                          # v11 B1: H10 min_bid_to_win = max(LU7, H1) on WON rows
                          # (SSP-full-feedback arm; C3/C4-only via censor)
                          "min_bid_to_win": False,
                          # v11 B2: exploration traffic slice (share of bids gets
                          # wide lognormal jitter; policy flag, no schema column)
                          "explore_traffic": False}
        _bn = self.benchmarks.get("bn", {}) or {}
        self.bn_edges = {**_edge_defaults, **(_bn.get("edges") or {})}
        bn_path = CONFIG_DIR / "bn_cpts.yaml"
        self.bn_cpts = (yaml.safe_load(bn_path.read_text(encoding="utf-8"))
                        if bn_path.exists() else None)

    # ── calibration shapes (iPinYou CSVs) ──────────────────────────
    def pmf(self, csv_name, value_col=None, weight_col="proportion"):
        """(values, probs) from a calibration CSV; probs renormalised."""
        df = pd.read_csv(CALIBRATION_DIR / csv_name)
        if value_col is None:
            value_col = df.columns[0]
        w = df[weight_col].to_numpy(dtype="float64")
        return df[value_col].to_numpy(), w / w.sum()

    def hourdow_pmf(self):
        """((dow, hour) value pairs, probs) from the hour x dow joint."""
        df = pd.read_csv(CALIBRATION_DIR /
                         self.benchmarks["time"]["hourdow_shape_csv"])
        w = df["proportion"].to_numpy(dtype="float64")
        return df[["dow", "hour"]].to_numpy(dtype="int64"), w / w.sum()

    def city_by_region(self):
        """{region: (city_values, probs)} for conditional A2 | A1 sampling."""
        df = pd.read_csv(CALIBRATION_DIR / "ipinyou_city_distribution.csv")
        region_col = df.columns[0]
        out = {}
        for region, grp in df.groupby(region_col):
            w = grp["proportion"].to_numpy(dtype="float64")
            out[region] = (grp["city"].to_numpy(), w / w.sum())
        return out

    def paying_median(self):
        s = pd.read_csv(CALIBRATION_DIR / "ipinyou_price_summary.csv",
                        index_col="series")
        return float(s.loc["paying_imp", "median"])

    # ── convenience ────────────────────────────────────────────────
    def rng(self, stream):
        """Independent, reproducible RNG per named module/stream."""
        digest = sum(ord(c) * 31**i for i, c in enumerate(stream)) % (2**31)
        return np.random.default_rng([int(self.profile["seed"]), digest])

    def output_dir(self):
        d = OUTPUT_DIR / self.profile_name
        d.mkdir(parents=True, exist_ok=True)
        return d


def split_shares(mapping):
    """{label: share} dict -> (labels list, probs array, renormalised)."""
    labels = list(mapping)
    p = np.array([mapping[k] for k in labels], dtype="float64")
    return labels, p / p.sum()
