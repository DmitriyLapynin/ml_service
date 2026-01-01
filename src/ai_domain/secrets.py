from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


class SecretNotFoundError(RuntimeError):
    pass


_DOTENV_CACHE: dict[Path, dict[str, str]] = {}


def _load_dotenv_file(path: Path) -> dict[str, str]:
    cached = _DOTENV_CACHE.get(path)
    if cached is not None:
        return cached
    if not path.exists() or not path.is_file():
        _DOTENV_CACHE[path] = {}
        return _DOTENV_CACHE[path]

    try:
        from dotenv import dotenv_values  # type: ignore[import-not-found]
    except Exception as e:  # pragma: no cover
        raise SecretNotFoundError(
            f"Secrets file {path} exists but python-dotenv is not installed; "
            "install it with `poetry add python-dotenv` or provide secrets via env vars."
        ) from e

    data = {k: (v or "") for k, v in dotenv_values(path).items() if k}
    _DOTENV_CACHE[path] = data
    return data


@dataclass(frozen=True)
class SecretsSource:
    """
    Production-like secrets loading:
    - Prefer environment variables (12-factor).
    - Fallback to a secrets `.env` file and/or individual secret files
      (e.g., docker/k8s mounted secrets).
    """

    secrets_dir: Path

    def get(self, name: str) -> Optional[str]:
        value = (os.getenv(name) or "").strip()
        if value:
            return value

        # Support a single secrets env-file (common for local/prod-like setups)
        # Example: ../secrets/.env containing SUPABASE_URL=... etc.
        dotenv_value = _load_dotenv_file(self.secrets_dir / ".env").get(name)
        if dotenv_value is not None:
            dotenv_value = dotenv_value.strip()
            if dotenv_value:
                return dotenv_value

        path = self.secrets_dir / name
        if not path.exists() or not path.is_file():
            return None

        return path.read_text(encoding="utf-8").strip()


def default_secrets_dir() -> Path:
    """
    Default location for external secrets:
      <repo_root>/../secrets

    Repo root is inferred from this file location:
      src/ai_domain/secrets.py -> repo_root = parents[2]
    """
    override = (os.getenv("AI_DOMAIN_SECRETS_DIR") or "").strip()
    if override:
        return Path(override).expanduser().resolve()

    repo_root = Path(__file__).resolve().parents[2]
    return (repo_root.parent / "secrets").resolve()


def get_secret(name: str, *, required: bool = False, secrets_dir: Path | None = None) -> Optional[str]:
    src = SecretsSource(secrets_dir=(secrets_dir or default_secrets_dir()))
    value = src.get(name)
    if required and not value:
        raise SecretNotFoundError(
            f"Missing secret {name}. Set env var {name} or create file {src.secrets_dir / name}"
        )
    return value
