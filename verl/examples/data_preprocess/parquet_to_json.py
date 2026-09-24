import argparse
import json
from pathlib import Path

import pandas as pd
import numpy as np


def to_jsonable(obj):
    """Recursively convert numpy/pandas objects to JSON-serializable Python objects."""
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [to_jsonable(v) for v in obj.tolist()]
    if isinstance(obj, np.generic):
        return obj.item()
    # handle NaN/NA
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj


def main():
    parser = argparse.ArgumentParser(description="Convert parquet file to json in the same directory.")
    parser.add_argument("parquet_path", type=str, help="Path to parquet file, e.g. /path/to/train.parquet")
    parser.add_argument(
        "--orient",
        type=str,
        default="records",
        choices=["records", "split", "index", "columns", "values", "table"],
        help="JSON orient format for pandas to_json",
    )
    args = parser.parse_args()

    parquet_path = Path(args.parquet_path).expanduser().resolve()
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

    df = pd.read_parquet(parquet_path)
    output_path = parquet_path.with_suffix(".json")

    if args.orient == "records":
        # Pretty JSON for manual inspection
        records = to_jsonable(df.to_dict(orient="records"))
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
    else:
        converted = to_jsonable(df.to_dict(orient="records"))
        if args.orient == "records":
            payload = converted
        elif args.orient == "table":
            payload = {"data": converted}
        else:
            # Fallback: keep a stable, readable structure even for uncommon orients.
            payload = converted
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    print(f"Saved: {output_path}")
    print(f"Rows: {len(df)}, Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
