import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from daytrader.config import load_config  # noqa: E402

CONFIG = Path(__file__).resolve().parents[1] / "config" / "spy_noise.yaml"


@pytest.fixture
def cfg():
    return load_config(CONFIG)
