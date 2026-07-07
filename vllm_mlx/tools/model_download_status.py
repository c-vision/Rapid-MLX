# SPDX-License-Identifier: Apache-2.0
"""Model download status manager - tracks download state for resume support."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


class ModelDownloadStatusManager:
    """Manage model download status tracking with atomic operations."""

    def __init__(self, cache_dir: Path | str | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "rapid-mlx" / "downloads"
        self.status_file = self.cache_dir / "download_status.json"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._init_status_file()

    def _init_status_file(self) -> None:
        """Initialize or load download status file."""
        if not self.status_file.exists():
            self._write_status({})

    def _read_status(self) -> Dict[str, Any]:
        """Read status file with error handling."""
        try:
            content = self.status_file.read_text()
            return {} if not content.strip() else __import__("json").loads(content)
        except Exception:
            return {}

    def _write_status(self, data: Dict[str, Any]) -> None:
        """Write status atomically to prevent corruption."""
        temp_file = self.status_file.with_suffix(".tmp")
        temp_file.write_text(__import__("json").dumps(data, indent=2))
        temp_file.replace(self.status_file)

    def get_model_status(self, model_name: str) -> Dict[str, Any]:
        """Get status for specific model."""
        status = self._read_status()
        return status.get(model_name, {})

    def set_model_status(
        self,
        model_name: str,
        status: str,
        path: Optional[str] = None,
        error: Optional[str] = None,
        last_byte: int = 0,
    ) -> None:
        """Set model status with validation."""
        status_data = self._read_status()
        status_data[model_name] = {
            "status": status,
            "timestamp": datetime.now().isoformat(),
            "path": path,
            "error": error,
            "attempts": status_data.get(model_name, {}).get("attempts", 0) + 1,
            "last_byte": max(0, int(last_byte)),
        }
        self._write_status(status_data)

    def is_model_downloaded(self, model_name: str) -> bool:
        """Check if model is fully downloaded."""
        status = self.get_model_status(model_name)
        return status.get("status") == "completed" and status.get("path")

    def get_download_progress(self, model_name: str) -> Dict[str, Any]:
        """Get detailed progress info."""
        status = self.get_model_status(model_name)
        if not status:
            return {"status": "not_started", "progress": 0, "downloaded_bytes": 0}

        path_str = status.get("path", "")
        last_byte = int(status.get("last_byte", 0))

        try:
            if path_str and Path(path_str).exists():
                file_size = Path(path_str).stat().st_size
                progress = (last_byte / file_size * 100) if file_size > 0 else 0
                return {
                    "status": status["status"],
                    "progress": min(100.0, progress),
                    "downloaded_bytes": last_byte,
                    "total_size": file_size,
                    "path": path_str,
                }
        except OSError:
            pass

        return {
            "status": status.get("status", "unknown"),
            "progress": status.get("progress", 0),
            "downloaded_bytes": last_byte,
        }