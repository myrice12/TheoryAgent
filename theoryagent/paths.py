"""Centralized runtime path helpers for TheoryAgent."""

from __future__ import annotations

import os
from pathlib import Path

THEORYAGENT_HOME_ENV = "THEORYAGENT_HOME"

_RUNTIME_SUBDIRS = (
    "workspace/research",
    "chat_memory",
    "cache/models",
    "cache/data",
)


def get_project_root() -> Path:
    """Return the repository root for the current editable install."""
    return Path(__file__).resolve().parent.parent


def get_legacy_runtime_home() -> Path:
    """Return the historical user-home runtime directory."""
    return Path.home() / ".theoryagent"


def get_runtime_home() -> Path:
    """Return the active runtime directory."""
    override = os.environ.get(THEORYAGENT_HOME_ENV, "").strip()
    if override:
        return Path(override).expanduser().resolve()
    return get_project_root() / ".theoryagent"


def ensure_runtime_home() -> Path:
    """Create the runtime directory layout if needed."""
    runtime_home = get_runtime_home()
    runtime_home.mkdir(parents=True, exist_ok=True)
    for subdir in _RUNTIME_SUBDIRS:
        (runtime_home / subdir).mkdir(parents=True, exist_ok=True)
    return runtime_home


def get_runtime_config_path() -> Path:
    """Return the primary config path under the active runtime home."""
    return get_runtime_home() / "config.json"


def resolve_config_path(config_path: Path | None = None) -> Path:
    """Resolve the config path for reads with legacy fallback."""
    if config_path is not None:
        return config_path.expanduser()

    runtime_config = get_runtime_config_path()
    if runtime_config.is_file():
        return runtime_config

    legacy_config = get_legacy_runtime_home() / "config.json"
    if legacy_config.is_file():
        return legacy_config

    return runtime_config


def get_workspace_root() -> Path:
    """Return the default workspace root."""
    return get_runtime_home() / "workspace" / "research"


def get_chat_memory_dir() -> Path:
    """Return the chat-memory directory."""
    return get_runtime_home() / "chat_memory"


def get_cache_root() -> Path:
    """Return the shared cache root."""
    return get_runtime_home() / "cache"
