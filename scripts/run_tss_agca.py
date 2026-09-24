#!/usr/bin/env python3
"""Run Refusal Sensitivity Attribution (TSS) and AGCA.

Stage 1 builds the attribution set and scores prompt tokens with TSS.
Stage 2 keeps the top-K refusal-sensitive tokens and synthesizes
intention-inverted benign queries.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts" / "attribution"))
sys.path.insert(0, str(ROOT / "scripts" / "agca"))

from build_attribution_set import run_attribution  # noqa: E402
from compute_tss import run_tss  # noqa: E402
from synthesize import run_agca  # noqa: E402


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="TSS attribution followed by AGCA synthesis.")
    p.add_argument("--skip_attribution", action="store_true")
    p.add_argument("--skip_tss", action="store_true")
    p.add_argument("--skip_agca", action="store_true")
    p.add_argument("--input", type=str, required=True, help="Harmful instruction JSON for the attribution set.")
    p.add_argument("--output_dir", type=str, default="", help="Run directory. Defaults to outputs/tss_agca/<time>.")
    p.add_argument("--attribution_output", type=str, default="")
    p.add_argument("--tss_output_dir", type=str, default="")
    p.add_argument("--agca_output", type=str, default="")

    p.add_argument("--generator_model", type=str, required=True, help="vLLM model used to sample refusal trajectories.")
    p.add_argument("--sample_attempts", type=int, default=5)
    p.add_argument("--instruction_field", type=str, default="instruction")
    p.add_argument("--attribution_max_samples", type=int, default=-1)
    p.add_argument("--attribution_temperature", type=float, default=0.6)
    p.add_argument("--gpu_memory_utilization", type=float, default=0.5)
    p.add_argument("--tensor_parallel_size", type=int, default=1)

    p.add_argument("--tss_model", type=str, required=True, help="HF model used for teacher-forced TSS.")
    p.add_argument("--tss_mode", type=str, default="all", choices=["A", "B", "C", "all"])
    p.add_argument("--mask_scope", type=str, default="prompt", choices=["query", "prompt"])
    p.add_argument("--only_safe_anchor", action="store_true")
    p.add_argument("--tss_max_samples", type=int, default=-1)
    p.add_argument("--kl_penalty", type=str, default="kl")
    p.add_argument(
        "--loss_agg_mode",
        type=str,
        default="token-mean",
        choices=["token-mean", "seq-mean-token-sum", "seq-mean-token-sum-norm", "seq-mean-token-mean"],
    )
    p.add_argument("--save_plot", action="store_true")
    p.add_argument("--plot_mode", type=str, default="sorted", choices=["sorted", "original"])
    p.add_argument("--mask_batch_size", type=int, default=32)
    p.add_argument("--mask_ngram_size", type=int, default=1)

    p.add_argument("--top_k", type=int, default=15, help="Top-K refusal-sensitive tokens kept by AGCA.")
    p.add_argument("--agca_model", type=str, default="gpt-4o-mini")
    p.add_argument("--agca_temperature", type=float, default=0.2)
    p.add_argument("--agca_max_retries", type=int, default=3)
    p.add_argument("--api_base", type=str, default="https://api.openai.com/v1")
    p.add_argument("--agca_max_workers", type=int, default=16)
    p.add_argument("--api_pool_max_error_count", type=int, default=5)
    p.add_argument("--api_pool_retry_minutes", type=int, default=30)
    p.add_argument("--api_pool_log_dir", type=str, default="")
    p.add_argument("--agca_max_samples", type=int, default=-1)
    p.add_argument("--resume", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = Path(args.output_dir) if args.output_dir else ROOT / "outputs" / "tss_agca" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)

    attribution_output = args.attribution_output or str(run_dir / "attribution_set.json")
    agca_output = args.agca_output or str(run_dir / "agca_contrastive.json")

    ngram = max(1, int(args.mask_ngram_size))
    scope_suffix = f"_{args.mask_scope}"
    if ngram > 1:
        scope_suffix = f"{scope_suffix}_ngram{ngram}"

    if args.skip_tss and not args.tss_output_dir.strip():
        if not args.agca_output.strip():
            raise ValueError("When skipping TSS, pass --tss_output_dir or --agca_output from an existing run.")
        tss_output_dir = str(Path(agca_output).parent / "tss")
    else:
        tss_output_dir = args.tss_output_dir or str(run_dir / "tss")
    tss_out_dir = Path(f"{tss_output_dir}{scope_suffix}")
    tss_json = tss_out_dir / "tss.json"

    if not args.skip_attribution:
        run_attribution(
            input_path=args.input,
            output_path=attribution_output,
            instruction_field=args.instruction_field,
            max_samples=args.attribution_max_samples,
            sample_attempts=args.sample_attempts,
            generator_model=args.generator_model,
            temperature=args.attribution_temperature,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
        )
    else:
        print("[skip] attribution set")

    if not args.skip_tss:
        run_tss(
            input_path=attribution_output,
            model_path=args.tss_model,
            output_dir=tss_output_dir,
            mode=args.tss_mode,
            mask_scope=args.mask_scope,
            only_safe_anchor=args.only_safe_anchor,
            max_samples=args.tss_max_samples,
            kl_penalty_type=args.kl_penalty,
            loss_agg_mode=args.loss_agg_mode,
            plot_mode=args.plot_mode,
            mask_batch_size=args.mask_batch_size,
            mask_ngram_size=ngram,
        )
    else:
        print("[skip] TSS")

    if not args.skip_agca:
        if not tss_json.is_file():
            raise FileNotFoundError(
                f"TSS file not found: {tss_json}. "
                "Pass --tss_output_dir as the directory given to compute_tss.py, without the mask-scope suffix."
            )
        run_agca(
            input_path=str(tss_json),
            output_path=agca_output,
            top_k=args.top_k,
            max_samples=args.agca_max_samples,
            model=args.agca_model,
            temperature=args.agca_temperature,
            max_retries=args.agca_max_retries,
            max_workers=args.agca_max_workers,
            resume=args.resume,
            api_base=args.api_base,
            api_pool_log_dir=args.api_pool_log_dir,
            api_pool_max_error_count=args.api_pool_max_error_count,
            api_pool_retry_minutes=args.api_pool_retry_minutes,
        )
    else:
        print("[skip] AGCA")

    print("Finished.")
    print(f"Run dir: {run_dir}")
    print(f"Attribution set: {attribution_output}")
    print(f"TSS: {tss_json}")
    print(f"AGCA: {agca_output}")


if __name__ == "__main__":
    main()
