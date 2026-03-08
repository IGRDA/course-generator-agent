"""Shared utilities for multi-API-key support across LLM providers.

Supports comma-separated API keys in environment variables.
A single key (no commas) works identically to a list of one.
"""

import os
import random
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests

_SECRETS_FILE = Path(__file__).resolve().parent.parent / "env.secrets"


def parse_api_keys(value: str | None) -> list[str]:
    """Parse a comma-separated string of API keys into a list.

    Args:
        value: Raw value from env var or kwarg, possibly comma-separated.

    Returns:
        List of stripped, non-empty API key strings.
        Empty list if value is None or empty.
    """
    if not value:
        return []
    return [k.strip() for k in value.split(",") if k.strip()]


def get_random_key(keys: list[str]) -> str:
    """Pick a random API key from the list.

    Args:
        keys: Non-empty list of API keys.

    Returns:
        A randomly selected key.
    """
    return random.choice(keys)


def mask_key(key: str) -> str:
    """Return a masked version of an API key for safe logging.

    Shows only the last 4 characters.

    Args:
        key: The full API key string.

    Returns:
        Masked string like ``...BvzA``.
    """
    if len(key) <= 4:
        return "***"
    return f"...{key[-4:]}"


def _load_all_keys_from_secrets(env_var: str) -> list[str]:
    """Read ALL keys for *env_var* from the env.secrets file on disk.

    Always reads from the file so that keys temporarily rate-limited in a
    previous run are retested.  Falls back to ``os.environ`` when the file
    does not exist (CI / Docker).
    """
    if _SECRETS_FILE.exists():
        text = _SECRETS_FILE.read_text()
        match = re.search(
            rf"^export\s+{re.escape(env_var)}=(.+)$", text, re.MULTILINE
        )
        if match:
            return parse_api_keys(match.group(1))

    return parse_api_keys(os.getenv(env_var, ""))


def _check_key(key: str, base_url: str, timeout: float) -> int:
    """Hit GET /v1/models to validate a single API key. Returns HTTP status."""
    try:
        resp = requests.get(
            f"{base_url}/models",
            headers={"Authorization": f"Bearer {key}"},
            timeout=timeout,
        )
        return resp.status_code
    except requests.Timeout:
        return 408
    except requests.RequestException:
        return 0


def validate_api_keys(
    env_var: str = "MISTRAL_API_KEY",
    base_url: str = "https://api.mistral.ai/v1",
    timeout: float = 10.0,
) -> list[str]:
    """Test ALL keys from env.secrets and set *env_var* to the healthy subset.

    Reads the full key list from ``env.secrets`` every time (never from
    ``os.environ``) so that keys that were rate-limited in a previous run
    are always retested.

    Raises ``RuntimeError`` when zero keys pass the health check.
    """
    all_keys = _load_all_keys_from_secrets(env_var)

    if not all_keys:
        raise RuntimeError(f"No API keys found for {env_var}")

    print(f"\n[APICheck] Testing {len(all_keys)} key(s) from {env_var}...")

    status_counts: Counter = Counter()
    healthy: list[str] = []
    results: dict[str, int] = {}

    with ThreadPoolExecutor(max_workers=len(all_keys)) as pool:
        futures = {
            pool.submit(_check_key, k, base_url, timeout): k for k in all_keys
        }
        for future in as_completed(futures):
            key = futures[future]
            status = future.result()
            results[key] = status

    for key in all_keys:
        status = results[key]
        status_counts[status] += 1
        label = "  OK " if status == 200 else " SKIP"
        print(f"  [{label}] Key {mask_key(key)} -> {status}")
        if status == 200:
            healthy.append(key)

    print(f"[APICheck] Status summary: {dict(sorted(status_counts.items()))}")
    print(f"[APICheck] Healthy keys: {len(healthy)}/{len(all_keys)}")

    if not healthy:
        raise RuntimeError(
            f"All {len(all_keys)} API keys failed health check. "
            f"Distribution: {dict(status_counts)}"
        )

    os.environ[env_var] = ",".join(healthy)
    return healthy
