# Copyright 2025
# Convert SFT/STAR-1/data/STAR-1.json (same as SFT/STAR-1 train/run_sft.sh --data_path) into
# verl MultiTurnSFTDataset parquet with a `messages` column: [user, assistant].
#
# Supports Qwen3-style <think>...</think> and dataset-style <redacted_thinking>...</redacted_thinking>.
# Usage:
#   python examples/data_preprocess/star1_to_verl_sft_parquet.py
#   python .../star1_to_verl_sft_parquet.py --harmful_json /path/to/STAR-1.json --output_dir data/star1_sft
#   python .../star1_to_verl_sft_parquet.py --benign 200 --harmful 200 --benign_json data/cc-benign.json
#
# --benign N / --harmful N: -1 keeps every row, 0 drops that side.
# When both sides are used, rows are interleaved by their count ratio.
# Writes data_source in {harmful, benign} so SFT CCR can apply only on harmful rows (model.ccr.only_harmful).

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any

import datasets

# Regex for inner thinking + remainder (after closing tag). Order: try think first, then redacted.
_RE_THINK = re.compile(r"<think>\s*(.*?)\s*</think>\s*(.*)", re.DOTALL)
_RE_REDACTED = re.compile(r"<redacted_thinking>\s*(.*?)\s*</redacted_thinking>\s*(.*)", re.DOTALL)


def _parse_thinking_and_attempt(response: str) -> tuple[str, str]:
    response = response.replace("\\/", "/").strip()
    m = _RE_THINK.search(response)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    m = _RE_REDACTED.search(response)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", response.strip()


def build_assistant_content(
    response: str,
    *,
    base_flag: int,
    think_flag: int,
    think_style: str = "auto",
) -> str:
    thinking_trajectory, attempt = _parse_thinking_and_attempt(response)

    if think_style == "auto":
        think_style = "think" if ("\u003c/think\u003e" in response) else "redacted"

    if think_flag:
        if think_style == "think":
            return f"<think>\n{thinking_trajectory}\n</think>\n\n{attempt}"
        return f"<redacted_thinking>\n{thinking_trajectory}\n</redacted_thinking>\n\n{attempt}"
    if base_flag:
        return attempt
    if think_style == "think":
        return "<think>\n\n</think>\n\n" + attempt
    return "<redacted_thinking>\n\n</redacted_thinking>\n\n" + attempt


def _parquet_metadata_str(val: Any) -> str | None:
    """Cast extra fields to str | None so mixed harmful/benign rows share one Arrow schema."""
    if val is None:
        return None
    if isinstance(val, (list, dict)):
        return json.dumps(val, ensure_ascii=False)
    return str(val)


def row_to_messages(example: dict, *, base_flag: int, think_flag: int, think_style: str) -> dict:
    q = example["question"]
    a = build_assistant_content(
        example["response"], base_flag=base_flag, think_flag=think_flag, think_style=think_style
    )
    ds = example.get("data_source")
    if not isinstance(ds, str) or not ds.strip():
        ds = "harmful"
    else:
        ds = ds.strip().lower()
        if ds not in ("harmful", "benign"):
            ds = "harmful"
    return {
        "messages": [
            {"role": "user", "content": q},
            {"role": "assistant", "content": a},
        ],
        "data_source": ds,
        "extra_id": _parquet_metadata_str(example.get("id")),
        "extra_category": _parquet_metadata_str(example.get("category")),
        "extra_source": _parquet_metadata_str(example.get("source")),
    }


def default_harmful_json(repo_verl_root: Path) -> Path:
    workspace_root = repo_verl_root.parent
    return workspace_root / "SFT" / "STAR-1" / "data" / "STAR-1.json"


def default_benign_json(repo_verl_root: Path) -> Path:
    return repo_verl_root / "data" / "cc-benign.json"


def load_harmful_json_array(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"Expected harmful JSON to be a list of objects: {path}")
    return raw


def load_benign_json(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        data = obj
    elif isinstance(obj, dict) and isinstance(obj.get("data"), list):
        data = obj["data"]
    else:
        raise ValueError(f"Unsupported benign JSON (need list or {{data: []}}): {path}")
    rows: list[dict[str, Any]] = []
    for r in data:
        if not isinstance(r, dict):
            continue
        q, resp = r.get("question"), r.get("response")
        if not isinstance(q, str) or not isinstance(resp, str):
            continue
        rows.append(
            {
                "question": q,
                "response": resp,
                "id": r.get("id"),
                "category": r.get("category", "benign"),
                "source": r.get("source", "cc-benign"),
            }
        )
    return rows


def take_up_to(items: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    """n==0 -> []; n<0 or n>=len -> all rows in file order; else a random subset of n."""
    if n == 0 or not items:
        return []
    if n < 0 or n >= len(items):
        return list(items)
    rng = random.Random(seed)
    seq = list(items)
    rng.shuffle(seq)
    return seq[:n]


def interleave_harmful_benign(
    harmful: list[dict[str, Any]], benign: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """
    Interleave harmful and benign rows by count ratio, not a fixed 1:1 alternation.
    80 harmful + 40 benign stays near 2:1; 100 + 40 stays near 5:2.
    """
    H, Bn = len(harmful), len(benign)
    if Bn == 0:
        return list(harmful)
    if H == 0:
        return list(benign)
    total = H + Bn
    out: list[dict[str, Any]] = []
    i = j = 0
    cw_h = 0
    cw_b = 0
    for _ in range(total):
        cw_h += H
        cw_b += Bn
        if cw_h >= cw_b:
            out.append(harmful[i])
            i += 1
            cw_h -= total
        else:
            out.append(benign[j])
            j += 1
            cw_b -= total
    return out


def _with_data_source(rows: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    return [{**r, "data_source": source} for r in rows]


def merge_harmful_benign_rows(
    harmful: list[dict[str, Any]],
    benign: list[dict[str, Any]],
    *,
    harmful_n: int,
    benign_n: int,
    seed: int,
) -> list[dict[str, Any]]:
    h_take = _with_data_source(take_up_to(harmful, harmful_n, seed), "harmful")
    b_take = _with_data_source(take_up_to(benign, benign_n, seed + 7919), "benign")
    if not h_take and not b_take:
        return []
    if not b_take:
        return h_take
    if not h_take:
        return b_take
    return interleave_harmful_benign(h_take, b_take)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--harmful_json",
        "--input_json",
        type=str,
        default=None,
        help="Harmful SFT JSON array. --input_json is an alias.",
    )
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--test_size", type=float, default=0.1)
    parser.add_argument("--train_max_samples", type=int, default=-1)
    parser.add_argument("--val_max_samples", type=int, default=-1)
    parser.add_argument("--base_flag", type=int, default=0)
    parser.add_argument("--think_flag", type=int, default=1)
    parser.add_argument(
        "--think_style",
        type=str,
        default="auto",
        choices=("auto", "think", "redacted"),
        help="Output tags: think=<think>; redacted=<redacted_thinking>; auto=from raw text.",
    )
    parser.add_argument(
        "--benign",
        type=int,
        default=0,
        metavar="N",
        help="Benign row count. -1 or a value >= the file length keeps all. 0 drops benign.",
    )
    parser.add_argument(
        "--harmful",
        type=int,
        default=-1,
        metavar="N",
        help="Harmful row count. -1 keeps all. 0 drops harmful.",
    )
    parser.add_argument(
        "--benign_json",
        type=str,
        default=None,
        help="Benign JSON path. Read only when --benign is not 0.",
    )
    args = parser.parse_args()

    repo_verl_root = Path(__file__).resolve().parents[2]
    harmful_json = Path(args.harmful_json) if args.harmful_json else default_harmful_json(repo_verl_root)
    output_dir = Path(args.output_dir) if args.output_dir else repo_verl_root / "data" / "star1_sft"
    output_dir = output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    benign_path = (
        Path(args.benign_json).expanduser().resolve()
        if args.benign_json
        else default_benign_json(repo_verl_root)
    )

    harmful_raw: list[dict[str, Any]] = []
    if args.harmful != 0:
        if not harmful_json.is_file():
            raise FileNotFoundError(f"Missing harmful JSON: {harmful_json}")
        harmful_raw = load_harmful_json_array(harmful_json)

    benign_raw: list[dict[str, Any]] = []
    if args.benign != 0:
        if not benign_path.is_file():
            raise FileNotFoundError(f"Missing benign JSON (--benign != 0): {benign_path}")
        benign_raw = load_benign_json(benign_path)

    merged_raw = merge_harmful_benign_rows(
        harmful_raw,
        benign_raw,
        harmful_n=args.harmful,
        benign_n=args.benign,
        seed=args.seed,
    )
    if not merged_raw:
        raise ValueError(
            "No rows left. Check that --harmful or --benign is non-zero and the JSON is non-empty."
        )

    rows = [
        row_to_messages(
            ex,
            base_flag=args.base_flag,
            think_flag=args.think_flag,
            think_style=args.think_style,
        )
        for ex in merged_raw
    ]
    ds = datasets.Dataset.from_list(rows)

    split = ds.train_test_split(seed=args.seed, test_size=args.test_size)
    train_ds, test_ds = split["train"], split["test"]

    if args.train_max_samples > 0 and args.train_max_samples < len(train_ds):
        train_ds = train_ds.shuffle(seed=args.seed).select(range(args.train_max_samples))
    if args.val_max_samples > 0 and args.val_max_samples < len(test_ds):
        test_ds = test_ds.shuffle(seed=args.seed).select(range(args.val_max_samples))

    train_path = output_dir / "train.parquet"
    val_path = output_dir / "test.parquet"
    train_ds.to_parquet(os.fspath(train_path))
    test_ds.to_parquet(os.fspath(val_path))

    print(f"Wrote {len(train_ds)} train / {len(test_ds)} val rows -> {output_dir}")
    print(f"  {train_path}")
    print(f"  {val_path}")


if __name__ == "__main__":
    main()
