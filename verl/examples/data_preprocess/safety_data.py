import argparse
import json
import os
import random
from pathlib import Path
from typing import Any

import datasets


def reward_data_source_from_json_type(data_type: str) -> str:
    """Map JSON data_type to verl reward data_source labels."""
    dt = (data_type or "safety")
    if not isinstance(dt, str):
        dt = str(dt)
    dt_lower = dt.strip().lower()
    if dt_lower in ("overreject", "benign", "harmless", "harmless_queries"):
        return "harmless_queries"
    # safety and unknown types use the harmful safety reward.
    return "custom_safety_dataset"


def _normalize_instruction(row: dict[str, Any]) -> str:
    text = row.get("instruction")
    if not isinstance(text, str) or not text.strip():
        text = row.get("question", "")
    if not isinstance(text, str):
        text = str(text)
    return text.strip()


def load_harmful_json(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if not isinstance(raw, list):
        raise ValueError(f"Expected harmful JSON to be a list of objects: {path}")
    rows: list[dict[str, Any]] = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        instruction = _normalize_instruction(r)
        if not instruction:
            continue
        rows.append(
            {
                "instruction": instruction,
                "data_type": r.get("data_type", "safety"),
                "risk_summary": r.get("risk_summary", ""),
                "difficulty": r.get("difficulty", ""),
            }
        )
    return rows


def load_benign_json(path: Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    if isinstance(raw, dict) and isinstance(raw.get("data"), list):
        raw = raw["data"]
    if not isinstance(raw, list):
        raise ValueError(f"Expected benign JSON to be list or {{data:[]}}: {path}")
    rows: list[dict[str, Any]] = []
    for r in raw:
        if not isinstance(r, dict):
            continue
        instruction = _normalize_instruction(r)
        if not instruction:
            continue
        rows.append(
            {
                "instruction": instruction,
                # overreject/benign rows map to harmless_queries for reward routing
                "data_type": r.get("data_type", "overreject"),
                "risk_summary": r.get("risk_summary", ""),
                "difficulty": r.get("difficulty", ""),
            }
        )
    return rows


def take_up_to(items: list[dict[str, Any]], n: int, seed: int) -> list[dict[str, Any]]:
    if n == 0 or not items:
        return []
    if n < 0 or n >= len(items):
        return list(items)
    seq = list(items)
    random.Random(seed).shuffle(seq)
    return seq[:n]


def interleave_by_ratio(harmful: list[dict[str, Any]], benign: list[dict[str, Any]]) -> list[dict[str, Any]]:
    h, b = len(harmful), len(benign)
    if h == 0:
        return list(benign)
    if b == 0:
        return list(harmful)
    total = h + b
    out: list[dict[str, Any]] = []
    i = j = 0
    cw_h = cw_b = 0
    for _ in range(total):
        cw_h += h
        cw_b += b
        if cw_h >= cw_b:
            out.append(harmful[i])
            i += 1
            cw_h -= total
        else:
            out.append(benign[j])
            j += 1
            cw_b -= total
    return out


def merge_harmful_benign_rows(
    harmful: list[dict[str, Any]], benign: list[dict[str, Any]], *, harmful_n: int, benign_n: int, seed: int
) -> list[dict[str, Any]]:
    h_take = take_up_to(harmful, harmful_n, seed)
    b_take = take_up_to(benign, benign_n, seed + 7919)
    if not h_take and not b_take:
        return []
    return interleave_by_ratio(h_take, b_take)


def _normalize_size(size: float | None, total: int) -> int | None:
    if size is None:
        return None
    if size <= 1:
        return int(round(total * size))
    return int(size)


def _compute_split_counts(total: int, train_size: float | None, val_size: float | None) -> tuple[int, int]:
    train_n = _normalize_size(train_size, total)
    test_n = _normalize_size(val_size, total)

    if train_n is None and test_n is None:
        test_n = int(round(total * 0.1))
    elif train_n is None:
        train_n = total - test_n
    elif test_n is None:
        test_n = total - train_n

    assert train_n is not None and test_n is not None
    if train_n < 0 or test_n < 0:
        raise ValueError(f"Invalid split sizes: train={train_n}, test={test_n}, total={total}")
    if train_n + test_n > total:
        raise ValueError(f"Requested train+test exceeds total: train={train_n}, test={test_n}, total={total}")
    return train_n, test_n


def _split_group(items: list[dict[str, Any]], test_n: int, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if test_n < 0 or test_n > len(items):
        raise ValueError(f"test_n out of range: test_n={test_n}, len={len(items)}")
    seq = list(items)
    random.Random(seed).shuffle(seq)
    return seq[test_n:], seq[:test_n]


def proportional_split_harmful_benign(
    harmful: list[dict[str, Any]],
    benign: list[dict[str, Any]],
    *,
    train_size: float | None,
    val_size: float | None,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    total = len(harmful) + len(benign)
    if total == 0:
        raise ValueError("No data to split.")

    train_n, test_n = _compute_split_counts(total, train_size, val_size)
    if test_n == 0:
        train_rows = interleave_by_ratio(harmful, benign)[:train_n]
        return train_rows, []

    harmful_test_n = int(round(test_n * (len(harmful) / total))) if harmful else 0
    harmful_test_n = max(0, min(harmful_test_n, len(harmful)))
    benign_test_n = test_n - harmful_test_n
    if benign_test_n > len(benign):
        spill = benign_test_n - len(benign)
        benign_test_n = len(benign)
        harmful_test_n = min(len(harmful), harmful_test_n + spill)
    if harmful_test_n > len(harmful):
        spill = harmful_test_n - len(harmful)
        harmful_test_n = len(harmful)
        benign_test_n = min(len(benign), benign_test_n + spill)
    if harmful_test_n + benign_test_n != test_n:
        raise ValueError(
            f"Unable to allocate test split proportionally: harmful_test={harmful_test_n}, "
            f"benign_test={benign_test_n}, test_n={test_n}"
        )

    harmful_train_pool, harmful_test = _split_group(harmful, harmful_test_n, seed)
    benign_train_pool, benign_test = _split_group(benign, benign_test_n, seed + 7919)

    train_rows = interleave_by_ratio(harmful_train_pool, benign_train_pool)[:train_n]
    test_rows = interleave_by_ratio(harmful_test, benign_test)
    return train_rows, test_rows


def make_map_fn(split: str, system_prompt: str = ""):
    """Insert a system turn before the user turn when system_prompt is non-empty."""

    sys_text = (system_prompt or "").strip()

    def process_fn(example, idx):
        instruction = example.get('instruction', '')
        data_type = example.get('data_type', 'safety')
        data_source = reward_data_source_from_json_type(data_type)

        if sys_text:
            prompt_messages = [
                {"role": "system", "content": sys_text},
                {"role": "user", "content": instruction},
            ]
        else:
            prompt_messages = [{"role": "user", "content": instruction}]

        data = {
            "data_source": data_source,
            "prompt": prompt_messages,
            "ability": data_type,
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "original_query": example.get("instruction", ""),
                }
            },
            "extra_info": {
                "split": split,
                "index": idx,
                "risk_summary": example.get('risk_summary', '') or '',
                "difficulty": example.get('difficulty', '') or '',
                "json_data_type": data_type,
            }
        }
        return data

    return process_fn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_json', '--harmful_json', dest='harmful_json', type=str, required=True)
    parser.add_argument('--benign_json', type=str, default=None)
    parser.add_argument('--harmful', type=int, default=-1, help='Number of harmful rows. -1 means all, 0 means none.')
    parser.add_argument('--benign', type=int, default=0, help='Number of benign rows. -1 means all, 0 means none.')
    parser.add_argument('--output_dir', '--local_dir', dest='local_dir', default=None)
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--train_data_size', type=float, default=None, help="Train split size, or a fraction in (0, 1).")
    parser.add_argument('--val_data_size', type=float, default=None, help="Validation split size, or a fraction in (0, 1).")
    parser.add_argument(
        "--system_prompt",
        type=str,
        default="",
        help="Optional system prompt. Empty means a user-only prompt.",
    )

    args = parser.parse_args()

    repo_verl_root = Path(__file__).resolve().parents[2]
    if args.local_dir is None:
        args.local_dir = str(repo_verl_root / "data" / "custom_safety_hb")

    harmful_path = Path(args.harmful_json).expanduser().resolve()
    if not harmful_path.is_file():
        raise FileNotFoundError(f"Missing harmful JSON: {harmful_path}")
    harmful_rows = load_harmful_json(harmful_path) if args.harmful != 0 else []

    benign_rows: list[dict[str, Any]] = []
    benign_path: Path | None = None
    if args.benign != 0:
        if not args.benign_json:
            raise ValueError("When --benign != 0, --benign_json is required.")
        benign_path = Path(args.benign_json).expanduser().resolve()
        if not benign_path.is_file():
            raise FileNotFoundError(f"Missing benign JSON: {benign_path}")
        benign_rows = load_benign_json(benign_path)

    h_take = take_up_to(harmful_rows, args.harmful, args.seed)
    b_take = take_up_to(benign_rows, args.benign, args.seed + 7919)
    if not h_take and not b_take:
        raise ValueError("No data after merge. Check --harmful/--benign and source files.")

    train_rows, test_rows = proportional_split_harmful_benign(
        h_take,
        b_take,
        train_size=args.train_data_size,
        val_size=args.val_data_size,
        seed=args.seed,
    )

    train_dataset = datasets.Dataset.from_list(train_rows)
    test_dataset = datasets.Dataset.from_list(test_rows)

    train_dataset = train_dataset.map(
        function=make_map_fn("train", args.system_prompt), with_indices=True
    )
    test_dataset = test_dataset.map(
        function=make_map_fn("test", args.system_prompt), with_indices=True
    )

    os.makedirs(args.local_dir, exist_ok=True)

    train_dataset.to_parquet(os.path.join(args.local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(args.local_dir, 'test.parquet'))

    out_dir = Path(args.local_dir)
    json_export_path = out_dir / "mixed_safety_dataset.json"
    export_payload = {
        "meta": {
            "harmful_json": str(harmful_path),
            "benign_json": str(benign_path) if benign_path is not None else None,
            "harmful_arg": args.harmful,
            "benign_arg": args.benign,
            "harmful_sampled": len(h_take),
            "benign_sampled": len(b_take),
            "seed": args.seed,
            "train_data_size": args.train_data_size,
            "val_data_size": args.val_data_size,
            "train_rows": len(train_dataset),
            "test_rows": len(test_dataset),
            "system_prompt_set": bool((args.system_prompt or "").strip()),
        },
        "train": train_dataset.to_list(),
        "test": test_dataset.to_list(),
    }
    with json_export_path.open("w", encoding="utf-8") as f:
        json.dump(export_payload, f, ensure_ascii=False, indent=2)

    print(f"Done. train={len(train_dataset)} val={len(test_dataset)} dir={args.local_dir}")
    print(f"JSON: {json_export_path}")

    if args.hdfs_dir is not None:
        from verl.utils.hdfs_io import copy, makedirs

        makedirs(args.hdfs_dir)
        copy(src=args.local_dir, dst=args.hdfs_dir)