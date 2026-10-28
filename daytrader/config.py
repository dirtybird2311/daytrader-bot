"""Load and validate run configuration."""

from __future__ import annotations

import copy
import datetime as dt
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    _validate(cfg)
    return cfg


def override(cfg: dict[str, Any], dotted: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of cfg with dotted keys replaced, e.g. {"params.allow_short": False}."""
    out = copy.deepcopy(cfg)
    for key, value in dotted.items():
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return out


def period_bounds(cfg: dict[str, Any], name: str) -> tuple[dt.date, dt.date | None]:
    """Start/end dates for 'tune', 'validate', 'holdout' or 'all'."""
    if name == "all":
        start = _date(cfg["data"]["start"])
        end = _date(cfg["data"].get("end"))
        return start, end
    if name not in cfg["periods"]:
        raise ValueError(f"Unknown period '{name}'. Use tune, validate, holdout or all.")
    start, end = cfg["periods"][name]
    return _date(start), _date(end)


def _date(value: Any) -> dt.date | None:
    if value is None:
        return None
    if isinstance(value, dt.date):
        return value
    return dt.date.fromisoformat(str(value))


def _validate(cfg: dict[str, Any]) -> None:
    required = ["strategy", "symbol", "data", "periods", "params", "sizing", "costs", "account"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"Config missing sections: {missing}")
    p = cfg["params"]
    if p["stop_check"] not in ("half_hour", "every_minute"):
        raise ValueError("params.stop_check must be 'half_hour' or 'every_minute'")
    if cfg["sizing"]["mode"] not in ("vol_target", "fixed"):
        raise ValueError("sizing.mode must be 'vol_target' or 'fixed'")
    if cfg["costs"]["fill_delay_bars"] < 1:
        raise ValueError("costs.fill_delay_bars must be >= 1 (no same-bar fills)")
    # Periods must be in time order and must not overlap.
    bounds = [period_bounds(cfg, n) for n in ("tune", "validate", "holdout")]
    for (s1, e1), (s2, _) in zip(bounds, bounds[1:]):
        if e1 is None or e1 >= s2:
            raise ValueError("periods must be in order tune < validate < holdout and must not overlap")
