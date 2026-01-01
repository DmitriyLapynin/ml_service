from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from ai_domain.secrets import get_secret


class SupabaseConfigError(RuntimeError):
    pass


@dataclass(frozen=True)
class SupabaseConfig:
    url: str
    key: str


def load_supabase_config_from_env(
    *,
    url_env: str = "SUPABASE_URL",
    key_env: str = "SUPABASE_KEY",
) -> SupabaseConfig:
    url = (get_secret(url_env) or "").strip()
    key = (get_secret(key_env) or "").strip()
    if not url:
        raise SupabaseConfigError(f"Missing env var {url_env}")
    if not key:
        raise SupabaseConfigError(f"Missing env var {key_env}")
    return SupabaseConfig(url=url, key=key)


def create_supabase_client(config: SupabaseConfig) -> Any:
    """
    Creates a `supabase-py` client.

    Requires `supabase` package:
      poetry add supabase

    Env var convention follows supabase-py docs:
      SUPABASE_URL, SUPABASE_KEY
    """
    try:
        from supabase import create_client  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise SupabaseConfigError(
            "supabase package is not installed; install it with `poetry add supabase`"
        ) from e

    return create_client(config.url, config.key)


def create_supabase_client_from_env(
    *,
    url_env: str = "SUPABASE_URL",
    key_env: str = "SUPABASE_KEY",
) -> Any:
    return create_supabase_client(load_supabase_config_from_env(url_env=url_env, key_env=key_env))
