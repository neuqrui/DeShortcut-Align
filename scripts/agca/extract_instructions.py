#!/usr/bin/env python3
"""Extract benign instructions from an AGCA JSON file."""
import argparse
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Extract benign instructions from an AGCA JSON file.")
    p.add_argument("--input", type=str, required=True, help="AGCA output JSON.")
    p.add_argument("--output", type=str, required=True, help="JSON list with an instruction field.")
    p.add_argument("--keep_empty", action="store_true", help="Keep rows whose benign instruction is empty.")
    return p.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and isinstance(data.get("data"), list):
        return data["data"]
    raise ValueError(f"Unsupported input format: {path}")


def main() -> None:
    args = parse_args()
    rows = load_rows(Path(args.input))
    out_rows = []
    for row in rows:
        text = row.get("benign_instruction") or row.get("instruction") or ""
        if not isinstance(text, str):
            text = str(text)
        text = text.strip()
        if not text and not args.keep_empty:
            continue
        out_rows.append({"instruction": text})
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(out_rows, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(out_rows)} instructions to {out_path}")


if __name__ == "__main__":
    main()
