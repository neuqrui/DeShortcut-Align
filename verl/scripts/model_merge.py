#!/usr/bin/env python3
"""Thin wrapper around legacy_model_merger merge.

Usage:
  python scripts/model_merge.py <local_dir> <target_dir> [--backend fsdp|megatron]

Example (run from verl root; PYTHONPATH must include the package root):
  python scripts/model_merge.py path/to/actor path/to/output_hf
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge verl FSDP/Megatron checkpoints to HuggingFace format (wraps legacy_model_merger merge).",
    )
    parser.add_argument("local_dir", help="Checkpoint directory (e.g. .../global_step_*/actor)")
    parser.add_argument("target_dir", help="Directory to write merged HuggingFace model")
    parser.add_argument(
        "--backend",
        default="fsdp",
        choices=["fsdp", "megatron"],
        help="Training backend used for the checkpoint (default: fsdp)",
    )
    args = parser.parse_args()

    scripts_dir = Path(__file__).resolve().parent
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))

    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "legacy_model_merger.py",
            "merge",
            "--backend",
            args.backend,
            "--local_dir",
            args.local_dir,
            "--target_dir",
            args.target_dir,
        ]
        from legacy_model_merger import main as merger_main

        merger_main()
    finally:
        sys.argv = old_argv


if __name__ == "__main__":
    main()
