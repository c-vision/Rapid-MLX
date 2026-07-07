# SPDX-License-Identifier: Apache-2.0
"""
Model downloader — fetches a full HuggingFace model repo (e.g. an MLX model,
which is a directory of config/tokenizer/weight-shard files, not a single
file) into a local directory, with resume support, parallel per-file
transfers, and SHA256 verification of LFS-tracked files.

Actual file transfer is delegated to huggingface_hub.snapshot_download
(already a core dependency) rather than reimplementing HTTP range-request
resume/retry logic — that's exactly what it's for, and it already handles
LFS redirects, partial-download resume, and per-file parallelism correctly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

import requests
from huggingface_hub import snapshot_download

from vllm_mlx.tools.model_download_status import ModelDownloadStatusManager


class ModelDownloadError(Exception):
    """Exception raised when a model download fails."""
    pass


class ModelDownloader:
    """Downloads full HuggingFace model repos with resume support and integrity verification."""

    MAX_CONCURRENT_DOWNLOADS = 8
    CONNECT_TIMEOUT = 10

    def __init__(self, cache_dir: Optional[Path | str] = None):
        """
        Initialize the model downloader.

        Args:
            cache_dir: Directory used to track download status (not where
                models themselves land — see `dest_dir` on download_model).
                Defaults to ~/.cache/rapid-mlx/downloads.
        """
        self.cache_dir = Path(cache_dir).expanduser() if cache_dir else Path.home() / ".cache" / "rapid-mlx" / "downloads"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Rapid-MLX-Downloader/1.0",
            "Accept": "*/*",
        })

        self.status_manager = ModelDownloadStatusManager(self.cache_dir)

    def _sha256_file(self, path: Path, chunk_size: int = 1024 * 1024) -> str:
        """Compute the SHA256 of a local file."""
        digest = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(chunk_size), b""):
                digest.update(chunk)
        return digest.hexdigest()

    def _fetch_repo_manifest(self, model_id: str) -> list[dict]:
        """
        Fetch the list of files (with size and, for LFS files, sha256) that
        HuggingFace records for this repo. Returns [] if the API call fails
        (caller falls back to a basic existence check in that case).
        """
        try:
            response = self.session.get(
                f"https://huggingface.co/api/models/{model_id}",
                params={"blobs": "true"},
                timeout=self.CONNECT_TIMEOUT,
            )
            response.raise_for_status()
            return response.json().get("siblings", [])
        except Exception:
            return []

    def _verify_repo_integrity(self, model_id: str, local_dir: Path) -> tuple[bool, list[str]]:
        """
        Verify downloaded files against HuggingFace's recorded metadata:
        SHA256 for LFS-tracked files (the weight shards), size for the rest.
        Returns (ok, list of problem descriptions).
        """
        siblings = self._fetch_repo_manifest(model_id)
        if not siblings:
            # Couldn't reach the manifest API — fall back to "did anything land".
            files = [f for f in local_dir.rglob("*") if f.is_file()]
            return (len(files) > 0), ([] if files else ["no files found after download"])

        problems: list[str] = []
        for sibling in siblings:
            filename = sibling.get("rfilename")
            if not filename:
                continue
            local_file = local_dir / filename
            if not local_file.exists():
                problems.append(f"{filename}: missing")
                continue

            lfs_info = sibling.get("lfs")
            expected_sha256 = lfs_info.get("sha256") if lfs_info else None
            if expected_sha256:
                actual_sha256 = self._sha256_file(local_file)
                if actual_sha256 != expected_sha256:
                    problems.append(f"{filename}: sha256 mismatch (expected {expected_sha256}, got {actual_sha256})")
            else:
                expected_size = sibling.get("size")
                if expected_size is not None and local_file.stat().st_size != expected_size:
                    problems.append(f"{filename}: size mismatch (expected {expected_size}, got {local_file.stat().st_size})")

        return (len(problems) == 0), problems

    def download_model(
        self,
        model_id: str,
        dest_dir: Optional[Path | str] = None,
        show_progress: bool = True,
    ) -> tuple[bool, Optional[str]]:
        """
        Download a full model repo, with resume support and SHA256
        verification of the downloaded weights.

        Args:
            model_id: HuggingFace repo ID, e.g. "mlx-community/Qwen3.5-27B-4bit"
            dest_dir: Parent directory to download into — the model lands in
                `dest_dir/<repo-name>` (matching how models are already laid
                out under e.g. ~/ai/Models). Defaults to this downloader's
                cache_dir if not given.
            show_progress: Whether to print progress/status messages.

        Returns:
            Tuple of (success, final directory path or None on failure).
        """
        target_root = Path(dest_dir).expanduser() if dest_dir else self.cache_dir
        local_name = model_id.split("/")[-1]
        final_dir = target_root / local_name

        # Status is tracked per model_id, but the same model can legitimately
        # be requested into different dest_dirs across calls — only treat it
        # as "already done" if the recorded completion matches THIS call's
        # target directory, not just any prior download of this model_id.
        status = self.status_manager.get_model_status(model_id)
        if (
            status.get("status") == "completed"
            and status.get("path") == str(final_dir)
            and final_dir.exists()
        ):
            if show_progress:
                print(f"✓ Model '{model_id}' already downloaded: {status['path']}")
            return True, status["path"]

        self.status_manager.set_model_status(model_id, "downloading")
        if show_progress:
            print(f"Downloading '{model_id}' into {final_dir} ...")

        try:
            snapshot_download(
                repo_id=model_id,
                local_dir=str(final_dir),
                max_workers=self.MAX_CONCURRENT_DOWNLOADS,
            )
        except Exception as e:
            self.status_manager.set_model_status(model_id, "error", error=str(e))
            if show_progress:
                print(f"✗ Download failed for '{model_id}': {e}")
            return False, None

        ok, problems = self._verify_repo_integrity(model_id, final_dir)
        if not ok:
            self.status_manager.set_model_status(model_id, "error", error="; ".join(problems))
            if show_progress:
                print(f"✗ Integrity check failed for '{model_id}':")
                for p in problems:
                    print(f"    - {p}")
            return False, None

        self.status_manager.set_model_status(model_id, "completed", str(final_dir))
        if show_progress:
            print(f"✓ Successfully downloaded '{model_id}' to: {final_dir}")
        return True, str(final_dir)


# Convenience function for external use
def download_model(
    model_id: str,
    dest_dir: Optional[Path | str] = None,
    show_progress: bool = True,
) -> tuple[bool, Optional[str]]:
    """
    Convenience function to download a model.

    Args:
        model_id: Model identifier (e.g., "mlx-community/Qwen3.5-27B-4bit")
        dest_dir: Parent directory to download into (model lands in
            dest_dir/<repo-name>). Defaults to the downloader's cache dir.
        show_progress: Whether to show progress messages.

    Returns:
        Tuple of (success: bool, final directory path or None on failure)
    """
    downloader = ModelDownloader()
    return downloader.download_model(model_id, dest_dir=dest_dir, show_progress=show_progress)
