import os
import time
import random
import asyncio
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime


def _load_dotenv() -> None:
    """Load key=value pairs from project-root .env into os.environ (no overwrite)."""
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if not env_path.is_file():
        return
    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _parse_api_keys(*env_names: str) -> List[str]:
    keys: List[str] = []
    for name in env_names:
        raw = os.environ.get(name, "").strip()
        if not raw:
            continue
        keys.extend(k.strip() for k in raw.split(",") if k.strip())
    # Deduplicate while preserving order.
    seen = set()
    unique: List[str] = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            unique.append(key)
    return unique


_load_dotenv()


class APIPool:
    def __init__(
        self,
        api_keys: List[str],
        pool_name: str = "default",
        max_error_count: int = 5,
        log_dir: Optional[str] = None,
        retry_interval_minutes: int = 30,
    ):
        self.pool_name = pool_name
        self.api_keys = api_keys
        self.max_error_count = max_error_count
        self.error_counts: Dict[str, int] = {key: 0 for key in api_keys}
        self.available_keys = set(api_keys)
        self.unavailable_timestamps: Dict[str, float] = {}
        self.retry_interval_seconds = retry_interval_minutes * 60

        self.log_dir = log_dir or os.path.join(os.getcwd(), "api_logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.log_file = os.path.join(
            self.log_dir, f"{pool_name}_pool_{datetime.now().strftime('%Y%m%d')}.log"
        )

    def _mask_api_key(self, api_key: str) -> str:
        if len(api_key) <= 8:
            return api_key
        return f"{api_key[:4]}...{api_key[-4:]}"

    def _log(self, message: str, is_error: bool = False) -> None:
        if is_error:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(f"[{timestamp}] [{self.pool_name}] {message}\n")

    def get_api_key(self) -> Optional[str]:
        current_time = time.time()
        keys_to_check = [
            k
            for k, t in self.unavailable_timestamps.items()
            if current_time - t >= self.retry_interval_seconds
        ]
        for k in keys_to_check:
            self.error_counts[k] = 0
            self.available_keys.add(k)
            self.unavailable_timestamps.pop(k, None)

        if not self.available_keys:
            self._log("Warning: No available API keys", is_error=True)
            return None
        return random.choice(list(self.available_keys))

    def mark_error(self, api_key: str, error_message: str) -> None:
        if api_key not in self.api_keys:
            return
        self.error_counts[api_key] += 1
        current_count = self.error_counts[api_key]
        self._log(
            f"API key {self._mask_api_key(api_key)} error "
            f"({current_count}/{self.max_error_count}): {error_message}",
            is_error=True,
        )
        if current_count >= self.max_error_count and api_key in self.available_keys:
            self.available_keys.remove(api_key)
            self.unavailable_timestamps[api_key] = time.time()


OPENAI_KEYS = _parse_api_keys("OPENAI_API_KEYS", "OPENAI_API_KEY")
openai_pool = APIPool(api_keys=OPENAI_KEYS, pool_name="openai")
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")

GEMINI_KEYS = _parse_api_keys("GEMINI_API_KEYS", "GEMINI_API_KEY")
gemini_pool = APIPool(api_keys=GEMINI_KEYS, pool_name="gemini")


async def _call_openai_async(prompt: str, model_name: str, temperature: float) -> str:
    from openai import AsyncOpenAI

    if not OPENAI_KEYS:
        return ""

    max_retries = min(3, len(openai_pool.available_keys) or 1)
    retry_count = 0

    while retry_count < max_retries:
        api_key = openai_pool.get_api_key()
        if api_key is None:
            return ""

        try:
            client = AsyncOpenAI(api_key=api_key, base_url=OPENAI_BASE_URL)
            response = await client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return response.choices[0].message.content or ""

        except Exception as e:
            error_message = str(e)
            openai_pool.mark_error(api_key, error_message)
            if "insufficient" in error_message.lower() or "rate_limit" in error_message.lower():
                await asyncio.sleep(2)
            retry_count += 1

    return ""


async def _call_gemini_async(
    prompt: str, model_name: str, temperature: float, thinking_level: str = "low"
) -> str:
    from google import genai
    from google.genai import types

    if not GEMINI_KEYS:
        return ""

    max_retries = min(3, len(gemini_pool.available_keys) or 1)
    retry_count = 0

    thinking_config = None
    if "gemini-3" in model_name.lower() or "gemini-2.0-pro-exp" in model_name.lower():
        thinking_config = types.ThinkingConfig(thinking_level=thinking_level)

    while retry_count < max_retries:
        api_key = gemini_pool.get_api_key()
        if api_key is None:
            return ""

        try:
            client = genai.Client(api_key=api_key)
            response = await client.aio.models.generate_content(
                model=model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=temperature,
                    thinking_config=thinking_config,
                ),
            )
            return response.text or ""

        except Exception as e:
            error_message = str(e)
            gemini_pool.mark_error(api_key, error_message)
            if "429" in error_message or "quota" in error_message.lower():
                await asyncio.sleep(2 * (retry_count + 1))
            retry_count += 1

    return ""


async def call_llm_judge_async(
    prompt: str,
    api_type: str = "openai",
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.1,
) -> str:
    if api_type == "openai":
        return await _call_openai_async(prompt, model_name, temperature)
    if api_type == "gemini":
        return await _call_gemini_async(prompt, model_name, temperature)
    raise ValueError(f"Unsupported api_type: {api_type}")
