# SPDX-License-Identifier: Apache-2.0
"""Tests for the alias_resolver module."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import yaml

from vllm_mlx.alias_resolver import (
    AliasConfig,
    AliasResolver,
    get_resolver,
    is_known_alias,
    resolve_alias,
    resolve_model_id,
    validate_alias_config,
)


class TestValidateAliasConfig:
    """Tests for configuration validation."""

    def test_valid_minimal_config(self):
        """Minimal valid config with just a model."""
        config = {"aliases": {"test": {"model": "mlx-community/Qwen3.5-4B-4bit"}}}
        errors = validate_alias_config(config)
        assert errors == []

    def test_valid_full_config(self):
        """Valid config with all fields."""
        config = {
            "aliases": {
                "full": {
                    "model": "mlx-community/Qwen3.5-4B-4bit",
                    "quant": "4bit",
                    "modality": "text",
                    "hardware": "m1",
                    "min_vram_gb": 4,
                    "path": "/custom/path",
                    "args": {"temp": 0.7},
                }
            }
        }
        errors = validate_alias_config(config)
        assert errors == []

    def test_missing_aliases_key(self):
        """Config must have top-level 'aliases' key."""
        config = {"models": {}}
        errors = validate_alias_config(config)
        assert len(errors) == 1
        assert "aliases" in errors[0]

    def test_missing_model_field(self):
        """Each alias must have a model field."""
        config = {"aliases": {"test": {"quant": "4bit"}}}
        errors = validate_alias_config(config)
        assert any("model" in e for e in errors)

    def test_invalid_quantization(self):
        """Invalid quant value should error."""
        config = {"aliases": {"test": {"model": "x", "quant": "invalid"}}}
        errors = validate_alias_config(config)
        assert any("quant" in e for e in errors)

    def test_invalid_modality(self):
        """Invalid modality value should error."""
        config = {"aliases": {"test": {"model": "x", "modality": "invalid"}}}
        errors = validate_alias_config(config)
        assert any("modality" in e for e in errors)

    def test_invalid_path_type(self):
        """Path must be a string if provided."""
        config = {"aliases": {"test": {"model": "x", "path": 123}}}
        errors = validate_alias_config(config)
        assert any("path" in e for e in errors)

    def test_empty_config(self):
        """Empty config should error."""
        errors = validate_alias_config(None)
        assert len(errors) == 1


class TestAliasConfig:
    """Tests for AliasConfig dataclass."""

    def test_from_dict(self):
        """Create AliasConfig from dictionary."""
        config = AliasConfig.from_dict({
            "model": "test/model",
            "quant": "4bit",
            "path": "/path",
        })
        assert config.model == "test/model"
        assert config.quant == "4bit"
        assert config.path == "/path"

    def test_to_dict(self):
        """Serialize AliasConfig to dictionary."""
        config = AliasConfig(
            model="test/model",
            quant="4bit",
            path="/path",
        )
        d = config.to_dict()
        assert d["model"] == "test/model"
        assert d["quant"] == "4bit"
        assert d["path"] == "/path"


class TestAliasResolver:
    """Tests for AliasResolver class."""

    @pytest.fixture
    def temp_config(self, tmp_path: Path):
        """Create a temporary config file."""
        config_path = tmp_path / "models.yaml"
        config_data = {
            "aliases": {
                "local": {
                    "model": "mlx-community/Qwen3.5-4B-4bit",
                    "path": "/custom/path",
                },
                "fast": {
                    "model": "mlx-community/Qwen3.5-4B-4bit",
                    "quant": "4bit",
                },
            }
        }
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)
        return config_path

    def test_resolve_user_alias(self, temp_config: Path):
        """Resolve a user-defined alias."""
        resolver = AliasResolver(temp_config)
        config = resolver.resolve("fast")
        assert config is not None
        assert config.model == "mlx-community/Qwen3.5-4B-4bit"
        assert config.quant == "4bit"

    def test_resolve_with_path_override(self, temp_config: Path):
        """Resolve alias with path override."""
        resolver = AliasResolver(temp_config)
        config = resolver.resolve("local")
        assert config is not None
        assert config.path == "/custom/path"

    def test_resolve_nonexistent_alias(self, temp_config: Path):
        """Nonexistent alias returns None."""
        resolver = AliasResolver(temp_config)
        config = resolver.resolve("nonexistent")
        assert config is None

    def test_resolve_model_id(self, temp_config: Path):
        """Resolve model ID from alias."""
        resolver = AliasResolver(temp_config)
        model_id = resolver.resolve_model("fast")
        assert model_id == "mlx-community/Qwen3.5-4B-4bit"

    def test_is_alias(self, temp_config: Path):
        """Check if a name is an alias."""
        resolver = AliasResolver(temp_config)
        assert resolver.is_alias("fast") is True
        assert resolver.is_alias("nonexistent") is False

    def test_list_aliases(self, temp_config: Path):
        """List all aliases."""
        resolver = AliasResolver(temp_config)
        aliases = resolver.list_aliases()
        assert "fast" in aliases
        assert "local" in aliases

    def test_get_model_dir_with_path(self, temp_config: Path):
        """Get model directory with path override."""
        resolver = AliasResolver(temp_config)
        model_dir = resolver.get_model_dir("local")
        assert model_dir == Path("/custom/path")

    def test_get_model_dir_without_path(self, temp_config: Path):
        """Get model directory without path override."""
        resolver = AliasResolver(temp_config)
        model_dir = resolver.get_model_dir("fast")
        # Should return None since fast has no path override
        assert model_dir is None


class TestSingletonResolver:
    """Tests for the singleton resolver functions."""

    def test_resolve_alias_function(self):
        """Test resolve_alias function."""
        # This will use the default resolver
        result = resolve_alias("nonexistent")
        assert result is None

    def test_resolve_model_id_function(self):
        """Test resolve_model_id function."""
        result = resolve_model_id("nonexistent")
        # Should return the input unchanged
        assert result == "nonexistent"

    def test_is_known_alias_function(self):
        """Test is_known_alias function."""
        result = is_known_alias("nonexistent")
        assert result is False


class TestPathValidation:
    """Tests for path field validation."""

    def test_valid_path(self):
        """Valid path string passes validation."""
        config = {"aliases": {"test": {"model": "x", "path": "/valid/path"}}}
        errors = validate_alias_config(config)
        assert not any("path" in e for e in errors)

    def test_empty_path_allowed(self):
        """Empty path is allowed (will be treated as None)."""
        config = {"aliases": {"test": {"model": "x", "path": ""}}}
        errors = validate_alias_config(config)
        # Empty string is still a string, so no error
        assert not any("path" in e for e in errors)

    def test_path_none_allowed(self):
        """None path is allowed."""
        config = {"aliases": {"test": {"model": "x", "path": None}}}
        errors = validate_alias_config(config)
        assert not any("path" in e for e in errors)