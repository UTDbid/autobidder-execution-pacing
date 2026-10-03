"""Config loader: profile loading, calibrated.yaml merge, RNG determinism."""
import numpy as np
import yaml

from t9sim.config_loader import Config
from t9sim.paths import CONFIG_DIR


def test_profiles_load():
    for p in ("golden", "test", "scale10m"):
        cfg = Config(p)
        assert int(cfg.profile["n_auctions"]) > 0
        assert int(cfg.profile["n_users"]) > 0


def test_calibrated_merge_applied():
    """Every solved knob in calibrated.yaml must override the base value and
    flip its provenance route to auto-calibrated."""
    cfg = Config("golden")
    solved = yaml.safe_load(
        (CONFIG_DIR / "calibrated.yaml").read_text(encoding="utf-8"))["solved"]
    assert solved, "calibrated.yaml has no solved knobs"
    for path, info in solved.items():
        keys = path.split(".")
        obj = getattr(cfg, keys[0])
        for k in keys[1:]:
            obj = obj[k]
        assert obj == info["value"], path
        row = cfg.provenance[cfg.provenance["param"] == path]
        assert (row["route"] == "auto-calibrated").all(), path


def test_rng_determinism_and_independence():
    cfg = Config("golden")
    a = cfg.rng("stream_a").random(8)
    a2 = cfg.rng("stream_a").random(8)
    b = cfg.rng("stream_b").random(8)
    np.testing.assert_array_equal(a, a2)          # same stream -> identical
    assert not np.array_equal(a, b)               # different stream -> differs


def test_provenance_registry_nonempty():
    cfg = Config("golden")
    prov = cfg.provenance
    assert len(prov) > 0
    assert set(["param", "value", "route", "source"]) <= set(prov.columns)
