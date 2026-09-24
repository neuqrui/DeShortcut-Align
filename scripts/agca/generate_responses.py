#!/usr/bin/env python3
"""Generate helpful responses for AGCA benign queries.

Sends only the user instruction. Joins reasoning_content and the final content
into one SFT response wrapped with think tags.
Fields: question, response, id, model.
Auth: --api_key, else DEEPSEEK_API_KEY, else OPENAI_API_KEY.

Example:
  python scripts/agca/generate_responses.py \\
    --input scripts/outputs/star-1/benign_instruction_only.json \\
    --output scripts/outputs/star-1/sft_benign_deepseek_official.json \\
    --max_samples 100 \\
    --num_workers 8
"""
from __future__ import annotations

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Multiprocess: benign instructions to helpful SFT responses."
    )
    p.add_argument("--input", type=str, required=True, help="Input JSON: a list or {data: [...]}.")
    p.add_argument("--output", type=str, required=True, help="Output JSON path.")
    p.add_argument("--instruction_field", type=str, default="instruction", help="Instruction field name.")
    p.add_argument(
        "--model",
        type=str,
        default="deepseek-v4-pro",
        help="Model name on the OpenAI-compatible endpoint.",
    )
    p.add_argument(
        "--api_base",
        type=str,
        default="https://api.deepseek.com",
        help="API base URL. Overridden by DEEPSEEK_BASE_URL.",
    )
    p.add_argument(
        "--api_key",
        type=str,
        default="",
        help="API key. Defaults to DEEPSEEK_API_KEY or OPENAI_API_KEY.",
    )
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--max_tokens", type=int, default=8192, help="<=0 omits max_tokens.")
    p.add_argument("--num_workers", type=int, default=8, help="Number of worker processes.")
    p.add_argument(
        "--max_samples",
        type=int,
        default=-1,
        metavar="N",
        help="Process only the first N rows. -1 means no limit.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="If the output exists, skip ids that are already written.",
    )
    return p.parse_args()


def resolve_api_key(cli_key: str) -> str:
    if cli_key.strip():
        return cli_key.strip()
    for env_name in ("DEEPSEEK_API_KEY", "OPENAI_API_KEY"):
        v = os.environ.get(env_name, "").strip()
        if v:
            return v
    return ""


def resolve_base_url(cli_base: str) -> str:
    if os.environ.get("DEEPSEEK_BASE_URL", "").strip():
        return os.environ["DEEPSEEK_BASE_URL"].strip().rstrip("/")
    return cli_base.strip().rstrip("/")


def load_dataset(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("data"), list):
        return obj["data"]
    raise ValueError(f"Unsupported input format: {path}")


def extract_instruction(row: Dict[str, Any], field: str) -> str:
    if field in row and isinstance(row[field], str):
        return row[field].strip()
    for k in ("instruction", "query", "prompt", "input"):
        if k in row and isinstance(row[k], str):
            return row[k].strip()
    raise KeyError(f"Instruction field {field!r} not found, keys={list(row.keys())}")


def load_done_ids(out_path: Path) -> set:
    if not out_path.exists():
        return set()
    with out_path.open("r", encoding="utf-8") as f:
        obj = json.load(f)
    rows = obj.get("data") if isinstance(obj, dict) else obj
    if not isinstance(rows, list):
        return set()
    done = set()
    for r in rows:
        if "id" in r and r["id"] is not None:
            done.add(int(r["id"]))
    return done


def split_chunks(items: List[Tuple[int, str]], n_workers: int) -> List[List[Tuple[int, str]]]:
    if not items:
        return []
    n_workers = max(1, min(n_workers, len(items)))
    chunks: List[List[Tuple[int, str]]] = [[] for _ in range(n_workers)]
    for i, it in enumerate(items):
        chunks[i % n_workers].append(it)
    return [c for c in chunks if c]


def build_response_key(reasoning_content: str, output_text: str) -> str:
    """Join reasoning_content and content into one SFT response string."""
    rc = (reasoning_content or "").strip()
    ot = (output_text or "").strip()
    return f"<think>\n{rc}\n</think>\n\n{ot}"


def deepseek_official_chat(
    api_key: str,
    base_url: str,
    model: str,
    user_text: str,
    temperature: float,
    max_tokens: int,
) -> Dict[str, Optional[str]]:
    """
    Single user turn. Returns content, reasoning_content, and error.
    """
    from openai import OpenAI

    last_err = ""
    for attempt in range(5):
        try:
            with OpenAI(api_key=api_key, base_url=base_url) as client:
                kwargs: Dict[str, Any] = {
                    "model": model,
                    "messages": [{"role": "user", "content": user_text}],
                    "temperature": temperature,
                    "extra_body": {"thinking": {"type": "enabled"}},
                }
                if max_tokens > 0:
                    kwargs["max_tokens"] = max_tokens
                response = client.chat.completions.create(**kwargs)
                msg = response.choices[0].message
                content = (msg.content or "").strip()
                reasoning = getattr(msg, "reasoning_content", None)
                reasoning_s = (reasoning or "").strip() if reasoning is not None else ""
                return {
                    "content": content,
                    "reasoning_content": reasoning_s,
                    "error": None,
                }
        except Exception as e:
            last_err = str(e)
            low = last_err.lower()
            if "rate" in low or "429" in last_err or "quota" in low:
                time.sleep(2 * (attempt + 1))
            else:
                time.sleep(1.0)
    return {"content": "", "reasoning_content": "", "error": last_err or "api_error"}


def _worker_process_chunk(
    chunk: List[Tuple[int, str]],
    api_key: str,
    base_url: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for idx, question in chunk:
        res = deepseek_official_chat(
            api_key=api_key,
            base_url=base_url,
            model=model,
            user_text=question,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        response = build_response_key(str(res.get("reasoning_content") or ""), str(res.get("content") or ""))
        row: Dict[str, Any] = {
            "id": idx,
            "question": question,
            "response": response,
            "model": model,
        }
        if res["error"]:
            row["error"] = res["error"]
        if not response.strip():
            row.setdefault("error", row.get("error") or "empty_response")
        out.append(row)
    return out


def main() -> None:
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    api_key = resolve_api_key(args.api_key)
    if not api_key:
        raise SystemExit(
            "No API key. Set DEEPSEEK_API_KEY or OPENAI_API_KEY, or pass --api_key."
        )
    base_url = resolve_base_url(args.api_base)

    rows = load_dataset(in_path)
    if args.max_samples > 0:
        rows = rows[: args.max_samples]

    done_ids: set = set()
    existing_data: List[Dict[str, Any]] = []
    if args.resume and out_path.exists():
        done_ids = load_done_ids(out_path)
        with out_path.open("r", encoding="utf-8") as f:
            old = json.load(f)
        existing_data = list(old.get("data", [])) if isinstance(old, dict) else list(old)

    pending: List[Tuple[int, str]] = []
    for i, row in enumerate(rows):
        if i in done_ids:
            continue
        inst = extract_instruction(row, args.instruction_field)
        if not inst:
            continue
        pending.append((i, inst))

    chunks = split_chunks(pending, args.num_workers)
    new_results: List[Dict[str, Any]] = []

    if chunks:
        with ProcessPoolExecutor(max_workers=len(chunks)) as ex:
            futs = [
                ex.submit(
                    _worker_process_chunk,
                    ch,
                    api_key,
                    base_url,
                    args.model,
                    args.temperature,
                    args.max_tokens,
                )
                for ch in chunks
            ]
            with tqdm(
                total=len(pending),
                desc="responses",
                unit="row",
                dynamic_ncols=True,
            ) as bar:
                for fut in as_completed(futs):
                    batch = fut.result()
                    new_results.extend(batch)
                    bar.update(len(batch))

    merged = {int(r["id"]): r for r in existing_data}
    for r in new_results:
        merged[int(r["id"])] = r
    final_list = [merged[k] for k in sorted(merged.keys())]

    payload = {
        "config": {
            "input": str(in_path),
            "output": str(out_path),
            "provider": "deepseek_official",
            "api_base": base_url,
            "model": args.model,
            "instruction_field": args.instruction_field,
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "num_workers": args.num_workers,
            "max_samples": args.max_samples,
            "resume": args.resume,
            "response_format": "<think>\\n{reasoning}\\n</think>\\n\\n{output}",
        },
        "summary": {
            "total_input_rows": len(rows),
            "generated_this_run": len(new_results),
            "total_output_rows": len(final_list),
        },
        "data": final_list,
    }
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print(
        f"Saved -> {out_path} | input={len(rows)} | this_run={len(new_results)} | total_out={len(final_list)}"
    )


if __name__ == "__main__":
    main()
