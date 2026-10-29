import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from daytrader.config import load_config, override  # noqa: E402

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
CONFIG = CONFIG_DIR / "spy_noise.yaml"

# Every single-symbol strategy goes through the same leak hunters.
SPY_VARIANTS = {
    "spy_noise": ("spy_noise.yaml", {}),
    "spy_noise_regime": ("spy_noise.yaml", {"params.regime_pct": 40, "params.regime_min_days": 20}),
    "spy_lasthalf": ("spy_lasthalf.yaml", {}),
    "spy_vwap_revert": ("spy_vwap_revert.yaml", {}),
}


@pytest.fixture(params=sorted(SPY_VARIANTS))
def cfg(request):
    name, changes = SPY_VARIANTS[request.param]
    c = load_config(CONFIG_DIR / name)
    return override(c, changes) if changes else c
