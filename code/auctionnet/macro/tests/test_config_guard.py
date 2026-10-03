import json
from pathlib import Path

import pytest

from platformbid_v1.evaluator import load_config


def test_formal_template_cannot_execute(tmp_path: Path):
    path = tmp_path / "formal.json"
    path.write_text(
        json.dumps(
            {
                "status": "TEMPLATE_NOT_LOCKED_DO_NOT_RUN",
                "alphas": [0, 1],
                "adopter_policies": [],
                "low_kappa": 0.1,
                "q_scale": 1,
                "h": 4,
                "reference_mode": "pacing",
                "lookback": 3,
                "max_action": 100,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(RuntimeError, match="CONFIG_NOT_LOCKED"):
        load_config(path)
