from .base import Intent, Strategy
from .spy_noise import SpyNoiseArea

STRATEGIES = {"spy_noise": SpyNoiseArea}


def build_strategy(cfg: dict) -> Strategy:
    name = cfg["strategy"]
    if name not in STRATEGIES:
        raise ValueError(f"Unknown strategy '{name}'. Available: {sorted(STRATEGIES)}")
    return STRATEGIES[name](cfg)
