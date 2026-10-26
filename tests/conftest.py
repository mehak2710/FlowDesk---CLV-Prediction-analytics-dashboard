import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from clv.data_generation import generate_tables  # noqa: E402


@pytest.fixture(scope="session")
def small_tables():
    """A small, fast, fully reproducible synthetic dataset."""
    return generate_tables(seed=7, n_customers=900)