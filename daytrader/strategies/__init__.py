from .base import Intent, Strategy
from .spy_lasthalf import SpyLastHalfHour
from .spy_noise import SpyNoiseArea
from .spy_vwap_revert import SpyVwapRevert

STRATEGIES = {"spy_noise": SpyNoiseArea, "spy_lasthalf": SpyLastHalfHour, "spy_vwap_revert": SpyVwapRevert}


def build_strategy(cfg: dict) -> Strategy:
    name = cfg["strategy"]
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{name}'. Available: {sorted(STRATEGIES)}")
    return STRATEGIES[name](cfg)
