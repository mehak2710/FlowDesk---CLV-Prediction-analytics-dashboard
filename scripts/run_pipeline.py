"""Train models, score customers and build the SQLite database used by the dashboard."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd  # noqa: E402

from clv import config as C  # noqa: E402
from clv.pipeline import run  # noqa: E402


def main() -> None:
    C.DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    result = run()
    r = result.report
    pd.set_option("display.width", 160)
    print("\nVALIDATION (train snapshots -> 2023-12-31)")
    print(pd.DataFrame({**r["validation_baselines"], **r["validation_metrics"]}).T.round(3))
    print("\nTEST (train+valid -> 2024-12-31, out-of-time)")
    print(pd.DataFrame({**r["test_baselines"], **r["test_metrics"]}).T.round(3))
    print("\nSegment thresholds:", json.dumps(r["segment_thresholds"]))
    print("\nChurn model:", json.dumps({k: r["churn_model"][k] for k in ("selected_model", "validation_auc", "test_auc", "test_brier", "test_base_rate")}))
    print("Churn tiers (backtest):", json.dumps(r["churn_model"]["tier_backtest"]))
    print("Risk validation:", json.dumps(r["risk_validation"]))


if __name__ == "__main__":
    main()