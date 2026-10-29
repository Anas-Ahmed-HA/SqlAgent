"""Loading `.env` before anything reads the environment.

``shared.settings`` freezes endpoint URLs into module constants at import
time, so call :func:`ensure_env_early` before importing it.
"""

from __future__ import annotations

import sys
from pathlib import Path

__all__ = ["app_dir_early", "ensure_env_early"]


def app_dir_early() -> Path:
    """Runtime state dir without importing ``shared.settings``."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parents[2]


def ensure_env_early() -> tuple[Path, Path, str]:
    """Load ``.env``, returning ``(app_dir, env_path, status)``.

    ``status`` is ``loaded`` or ``missing``. The real process environment
    stays above the file (``override=False``).
    """
    from dotenv import load_dotenv

    app_dir = app_dir_early()
    env_path = app_dir / ".env"
    status = "loaded" if load_dotenv(env_path) else "missing"
    return app_dir, env_path, status
