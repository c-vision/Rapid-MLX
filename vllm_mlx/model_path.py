# SPDX-License-Identifier: Apache-2.0
"""
Model Path Resolution for rapid-mlx.

Provides configurable model storage directory resolution.

Environment variables:
    RAPID_MLX_MODEL_PATH - Override model storage directory

Fallback order:
    1. RAPID_MLX_MODEL_PATH
    2. ~/.cache/rapid-mlx/models
    3. default HuggingFace cache
"""

from __future__ import annotations

import os
from pathlib import Path

from huggingface_hub.constants import HF_HUB_CACHE


# Default paths
RAPID_MLX_CACHE = Path.home() / ".cache" / "rapid-mlx" / "models"

# Environment variable name
MODEL_PATH_ENV = "RAPID_MLX_MODEL_PATH"


def get_model_path() -> Path:
    """
    Get the model storage directory.

    Resolution order:
        1. RAPID_MLX_MODEL_PATH (if set)
        2. ~/.cache/rapid-mlx/models
        3. HuggingFace cache

    Returns:
        Path to the model storage directory
    """
    env_path = os.environ.get(MODEL_PATH_ENV, "").strip()
    if env_path:
        return Path(env_path).expanduser()

    return RAPID_MLX_CACHE


def get_cache_search_paths() -> list[Path]:
    """
    Get the ordered list of cache directories to search.

    Returns:
        List of cache directory paths in priority order
    """
    paths = []

    env_path = os.environ.get(MODEL_PATH_ENV, "").strip()
    if env_path:
        paths.append(Path(env_path).expanduser())

    paths.append(RAPID_MLX_CACHE)
    paths.append(Path(HF_HUB_CACHE))
    return paths


def resolve_model_dir(repo_id: str) -> Path | None:
    """
    Find an existing model directory for a given repo ID.

    Searches through the cache paths in priority order and returns
    the first directory that contains the model.

    Args:
        repo_id: The HuggingFace repo ID (e.g., 'mlx-community/Qwen3.5-4B')

    Returns:
        Path to the model directory if found, None otherwise
    """
    from huggingface_hub import try_to_load_from_cache

    # Try each cache path
    for cache_root in get_cache_search_paths():
        try:
            # Let HF resolve the path using its own cache logic
            result = try_to_load_from_cache(
                repo_id=repo_id,
                filename="config.json",
                cache_dir=str(cache_root),
            )
            if result is not None and Path(result).exists():
                # Return the parent directory of config.json
                return Path(result).parent
        except Exception:
            continue

    return None


def ensure_model_dir() -> Path:
    """
    Ensure the model storage directory exists.

    Returns:
        Path to the model storage directory (created if necessary)
    """
    path = get_model_path()
    path.mkdir(parents=True, exist_ok=True)
    return path


def configure_model_dir(path: str | Path) -> None:
    """
    Configure a custom model directory via environment variable.

    Note: This only affects the current process. For persistent
    configuration, update the environment or shell profile.

    Args:
        path: The new model directory path
    """
    os.environ[MODEL_PATH_ENV] = str(Path(path).expanduser())