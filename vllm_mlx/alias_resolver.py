# SPDX-License-Identifier: Apache-2.0
"""
Alias Resolver for rapid-mlx.

Maps short aliases (e.g. 'fast', 'coder') to full HF model IDs.

Supports two sources:
1. User-configurable aliases via ~/.rapid-mlx/models.yaml
2. Built-in aliases via vllm_mlx.model_aliases (aliases.json)

User configuration file: ~/.rapid-mlx/models.yaml

Example:
    aliases:
      fast:
        model: mlx-community/Qwen3.5-4B
        quant: 4bit
      coder:
        model: Qwen/Qwen3.5-Coder-27B
        quant: 8bit
"""

from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, List

import yaml


# Default path for the user alias configuration file
DEFAULT_CONFIG_PATH = Path.home() / ".rapid-mlx" / "models.yaml"


# ---------------------------------------------------------------------------
# Configuration validation utilities
# ---------------------------------------------------------------------------

# Allowed values and patterns for configuration fields
_ALLOWED_QUANTIZATION = {"4bit", "8bit", "3bit", "bpw"}
_ALLOWED_MODALITY = {"text", "vision", "audio"}
_ALLOWED_HARDWARE = {"m1", "m2", "m3", "m4", "m5", "m6", "m7", "m8", "m9", "m10"}
_VRAM_PATTERN = re.compile(r"^\d+$")


def _validate_quantization(value) -> bool:
    """Check that quantization value is one of the allowed types (or None)."""
    if value is None:
        return True
    return value in _ALLOWED_QUANTIZATION


def _validate_modality(value) -> bool:
    """Check that modality value is one of the allowed types (or None)."""
    if value is None:
        return True
    return value in _ALLOWED_MODALITY


def _validate_hardware(value) -> bool:
    """Check that hardware value is one of the allowed hardware identifiers (or None)."""
    if value is None:
        return True
    return value in _ALLOWED_HARDWARE


def _validate_vram_gb(value) -> bool:
    """Check that min_vram_gb is a positive integer (or None)."""
    if value is None:
        return True
    return isinstance(value, int) and value > 0 and bool(_VRAM_PATTERN.match(str(value)))


def validate_alias_config(config_data: dict) -> List[str]:
    """
    Validate a parsed alias configuration dictionary.

    Args:
        config_data: Dictionary representing the parsed YAML alias config.

    Returns:
        List of validation error messages. Empty list indicates the config is valid.
    """
    errors: List[str] = []

    if config_data is None:
        return ["Configuration is empty or could not be parsed."]

    # Ensure the top-level structure has an 'aliases' key
    if "aliases" not in config_data:
        errors.append("Configuration must contain a top-level 'aliases' key.")
        return errors

    if not isinstance(config_data["aliases"], dict):
        errors.append("'aliases' must be a dictionary of alias name → config.")
        return errors

    # Validate each alias entry
    for alias_name, alias_config in config_data["aliases"].items():
        if not isinstance(alias_name, str) or not alias_name:
            errors.append(f"Alias name must be a non-empty string: '{alias_name}'")
            continue

        if not isinstance(alias_config, dict):
            errors.append(f"Alias '{alias_name}' must have a dictionary value.")
            continue

        # Required field
        if "model" not in alias_config:
            errors.append(f"Alias '{alias_name}' is missing the required 'model' field.")
        elif not isinstance(alias_config["model"], str) or not alias_config["model"]:
            errors.append(f"Alias '{alias_name}' 'model' must be a non-empty string.")

        # Optional fields with validation
        if "quant" in alias_config and not _validate_quantization(alias_config["quant"]):
            errors.append(
                f"Alias '{alias_name}' 'quant' must be one of {sorted(_ALLOWED_QUANTIZATION)}."
            )

        if "modality" in alias_config and not _validate_modality(alias_config["modality"]):
            errors.append(
                f"Alias '{alias_name}' 'modality' must be one of {sorted(_ALLOWED_MODALITY)}."
            )

        if "hardware" in alias_config and not _validate_hardware(alias_config["hardware"]):
            errors.append(
                f"Alias '{alias_name}' 'hardware' must be one of {sorted(_ALLOWED_HARDWARE)}."
            )

        if "min_vram_gb" in alias_config and not _validate_vram_gb(alias_config["min_vram_gb"]):
            errors.append(f"Alias '{alias_name}' 'min_vram_gb' must be a positive integer.")

        # Validate args dictionary entries (optional)
        if "args" in alias_config:
            if not isinstance(alias_config["args"], dict):
                errors.append(f"Alias '{alias_name}' 'args' must be a dictionary.")
            else:
                for arg_key in alias_config["args"]:
                    if not isinstance(arg_key, str):
                        errors.append(
                            f"Alias '{alias_name}' 'args' key '{arg_key}' must be a string."
                        )

    return errors


class AliasConfig:
    """Configuration container for a single alias."""

    def __init__(
        self,
        model: str,
        quant: str | None = None,
        tokenizer_override: str | None = None,
        args: dict[str, Any] | None = None,
    ):
        self.model = model
        self.quant = quant
        self.tokenizer_override = tokenizer_override
        self.args = args or {}

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary for YAML output."""
        result: dict[str, Any] = {"model": self.model}
        if self.quant:
            result["quant"] = self.quant
        if self.tokenizer_override:
            result["tokenizer_override"] = self.tokenizer_override
        if self.args:
            result["args"] = self.args
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AliasConfig":
        """Create from dictionary."""
        return cls(
            model=data["model"],
            quant=data.get("quant"),
            tokenizer_override=data.get("tokenizer_override"),
            args=data.get("args"),
        )


class AliasResolver:
    """
    Resolves short aliases to full model configurations.

    Priority order:
    1. User aliases from ~/.rapid-mlx/models.yaml
    2. Built-in aliases from vllm_mlx.model_aliases (aliases.json)

    Loads configuration from ~/.rapid-mlx/models.yaml and resolves
    alias names to their corresponding model settings.
    """

    def __init__(self, config_path: Path | str | None = None):
        self.config_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._user_aliases: dict[str, AliasConfig] = {}
        self._loaded = False
        self._validation_errors: List[str] = []

    def _load_user_config(self) -> None:
        """Load the user configuration file if not already loaded."""
        if self._loaded:
            return

        if self.config_path.exists():
            try:
                with open(self.config_path, "r") as f:
                    data = yaml.safe_load(f)

                # Validate the configuration before applying it.
                self._validation_errors = validate_alias_config(data or {})
                if self._validation_errors:
                    for err in self._validation_errors:
                        print(f"[alias_resolver] config error: {err}", file=sys.stderr)

                if data and "aliases" in data:
                    for alias, config in data["aliases"].items():
                        if isinstance(config, dict) and "model" in config:
                            self._user_aliases[alias] = AliasConfig.from_dict(config)
            except Exception:
                # Silently fail if config is malformed - don't block CLI
                pass

        self._loaded = True

    def get_validation_errors(self) -> List[str]:
        """Return validation errors encountered while loading the config."""
        self._load_user_config()
        return self._validation_errors

    def resolve(self, alias: str) -> AliasConfig | None:
        """
        Resolve an alias to its configuration.

        Args:
            alias: The alias name to resolve (e.g., 'fast', 'coder')

        Returns:
            AliasConfig if found, None otherwise
        """
        self._load_user_config()

        # First check user aliases
        if alias in self._user_aliases:
            return self._user_aliases[alias]

        # Then check built-in aliases
        try:
            from vllm_mlx.model_aliases import resolve_profile

            profile = resolve_profile(alias)
            if profile is not None:
                # Convert AliasProfile to AliasConfig
                return AliasConfig(
                    model=profile.hf_path,
                    quant=None,  # Quant not available in AliasProfile
                    tokenizer_override=None,
                    args={},
                )
        except Exception:
            pass

        return None

    def resolve_model(self, alias: str) -> str | None:
        """
        Resolve an alias to just the model ID.

        Args:
            alias: The alias name to resolve

        Returns:
            The model ID string if found, None otherwise
        """
        config = self.resolve(alias)
        return config.model if config else None

    def is_alias(self, name: str) -> bool:
        """Check if a name is a known alias."""
        self._load_user_config()

        # Check user aliases first
        if name in self._user_aliases:
            return True

        # Then check built-in aliases
        try:
            from vllm_mlx.model_aliases import is_known_alias as builtin_is_alias
            return builtin_is_alias(name)
        except Exception:
            return False

    def list_aliases(self) -> list[str]:
        """List all available aliases (user + built-in)."""
        self._load_user_config()

        # Get user aliases
        aliases = set(self._user_aliases.keys())

        # Add built-in aliases
        try:
            from vllm_mlx.model_aliases import list_aliases
            aliases.update(list_aliases().keys())
        except Exception:
            pass

        return sorted(aliases)

    def get_all_configs(self) -> dict[str, AliasConfig]:
        """Get all alias configurations (user + built-in)."""
        self._load_user_config()

        result = dict(self._user_aliases)

        # Add built-in aliases
        try:
            from vllm_mlx.model_aliases import resolve_profile

            for alias in list_aliases().keys():
                if alias not in result:
                    profile = resolve_profile(alias)
                    if profile is not None:
                        result[alias] = AliasConfig(
                            model=profile.hf_path,
                            quant=None,
                            tokenizer_override=None,
                            args={},
                        )
        except Exception:
            pass

        return result


# Global singleton instance
_resolver: AliasResolver | None = None


def get_resolver(config_path: Path | str | None = None) -> AliasResolver:
    """Get or create the global alias resolver instance."""
    global _resolver
    if _resolver is None:
        _resolver = AliasResolver(config_path)
    return _resolver


def resolve_alias(name: str) -> AliasConfig | None:
    """
    Convenience function to resolve an alias.

    Args:
        name: The alias or model ID to resolve

    Returns:
        AliasConfig if name is an alias, None if it's not a known alias
    """
    return get_resolver().resolve(name)


def resolve_model_id(name: str) -> str:
    """
    Resolve a name to a model ID.

    If the name is a known alias, returns the model ID from the alias config.
    Otherwise, returns the name unchanged (backward compatible behavior).

    Args:
        name: The alias or model ID

    Returns:
        The resolved model ID
    """
    config = resolve_alias(name)
    if config:
        return config.model
    return name


def get_model_quant(name: str) -> str | None:
    """
    Get the quantization setting for an alias.

    Args:
        name: The alias name

    Returns:
        Quantization string (e.g., '4bit', '8bit') or None
    """
    config = resolve_alias(name)
    return config.quant if config else None


def is_known_alias(name: str) -> bool:
    """Check if a name is a known alias."""
    return get_resolver().is_alias(name)
