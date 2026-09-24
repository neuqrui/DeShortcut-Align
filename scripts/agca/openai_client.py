"""Synchronous OpenAI Chat Completions client for AGCA.

Reuses reward/llm_api.APIPool for key rotation. Does not call the async judge APIs.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, List, Optional, Tuple

_llm_api_mod: Any = None


def _load_llm_api_module() -> Any:
    global _llm_api_mod
    if _llm_api_mod is not None:
        return _llm_api_mod
    try:
        from verl.reward import llm_api as m  # type: ignore
    except ImportError:
        repo_root = Path(__file__).resolve().parents[2]
        repo_verl = repo_root / "verl"
        if str(repo_verl) not in sys.path:
            sys.path.insert(0, str(repo_verl))
        try:
            import importlib

            m = importlib.import_module("reward.llm_api")
        except ImportError:
            import importlib.util

            llm_api_path = repo_verl / "reward" / "llm_api.py"
            spec = importlib.util.spec_from_file_location("reward_llm_api", str(llm_api_path))
            if spec is None or spec.loader is None:
                raise ImportError(f"Cannot load llm_api from {llm_api_path}")
            m = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(m)
    _llm_api_mod = m
    return m


def collect_openai_keys_from_env() -> List[str]:
    keys: List[str] = []
    raw = os.environ.get("OPENAI_API_KEYS", "").strip()
    if raw:
        keys.extend(k.strip() for k in raw.split(",") if k.strip())
    one = os.environ.get("OPENAI_API_KEY", "").strip()
    if one and one not in keys:
        keys.append(one)
    return keys


def build_openai_pool(
    log_dir: str = "",
    max_error_count: int = 5,
    retry_interval_minutes: int = 30,
) -> Tuple[Any, str]:
    """Return (APIPool, default base URL). Requires OPENAI_API_KEY or OPENAI_API_KEYS."""
    mod = _load_llm_api_module()
    APIPool = mod.APIPool
    default_base = getattr(mod, "OPENAI_BASE_URL", "https://api.openai.com/v1")
    keys = collect_openai_keys_from_env()
    if keys:
        ld = log_dir.strip() or None
        pool = APIPool(
            api_keys=keys,
            pool_name="agca_openai",
            max_error_count=max_error_count,
            log_dir=ld,
            retry_interval_minutes=retry_interval_minutes,
        )
        return pool, default_base
    raise RuntimeError("Set OPENAI_API_KEY or OPENAI_API_KEYS before running AGCA.")


def resolve_openai_base_url(cli_base: str, mod_default: str) -> str:
    if os.environ.get("OPENAI_BASE_URL", "").strip():
        return os.environ["OPENAI_BASE_URL"].strip().rstrip("/")
    if cli_base.strip():
        return cli_base.strip().rstrip("/")
    return mod_default.rstrip("/")


def openai_chat_completion_user_text(
    pool: Any,
    base_url: str,
    model: str,
    user_text: str,
    temperature: float,
    max_tokens: Optional[int] = None,
) -> str:
    """One user message, synchronous, rotating keys on the pool."""
    from openai import OpenAI

    max_retries = min(3, len(getattr(pool, "api_keys", [])) or 1)
    retry_count = 0
    while retry_count < max_retries:
        api_key = pool.get_api_key()
        if api_key is None:
            return ""
        try:
            with OpenAI(api_key=api_key, base_url=base_url) as client:
                kwargs = {
                    "model": model,
                    "messages": [{"role": "user", "content": user_text}],
                    "temperature": temperature,
                }
                if max_tokens is not None:
                    kwargs["max_tokens"] = max_tokens
                response = client.chat.completions.create(**kwargs)
                return (response.choices[0].message.content or "").strip()
        except Exception as e:
            err = str(e)
            pool.mark_error(api_key, err)
            if "insufficient" in err.lower() or "rate_limit" in err.lower():
                time.sleep(2)
            retry_count += 1
    return ""
