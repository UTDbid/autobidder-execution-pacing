"""Shared fixtures. The golden master is a small fixed-seed in-memory run on
an independent RNG stream ('pytest'); invariants must hold for any stream."""
import pytest

from t9sim import user_profiles
from t9sim.auctions import AuctionEngine
from t9sim.catalogue import build_apps, build_campaigns
from t9sim.config_loader import Config

N_TEST = 60_000


@pytest.fixture(scope="session")
def master():
    cfg = Config("golden")
    users = user_profiles.generate_users(cfg)
    engine = AuctionEngine(cfg, users, build_apps(cfg), build_campaigns(cfg))
    engine.calibrate()
    df = engine.chunk(N_TEST, cfg.rng("pytest"))
    return cfg, df
