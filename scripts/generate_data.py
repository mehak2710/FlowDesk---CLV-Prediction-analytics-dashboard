"""Generate the synthetic Flowdesk dataset, validate it, and write CSVs to data/raw/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from clv import config as C  # noqa: E402
from clv.data_generation import generate_tables, write_tables  # noqa: E402
from clv.dictionary import render_markdown  # noqa: E402
from clv.validation import assert_valid, format_report  # noqa: E402


def main() -> None:
    tables = generate_tables(seed=C.SEED, n_customers=C.N_CUSTOMERS)
    results = assert_valid(tables)
    write_tables(tables)
    (C.ROOT / "docs").mkdir(exist_ok=True)
    (C.ROOT / "docs" / "data_dictionary.md").write_text(render_markdown())
    print(format_report(results))
    print()
    for name, df in tables.items():
        print(f"{name:20s} {len(df):>8,d} rows -> data/raw/{name}.csv")


if __name__ == "__main__":
    main()