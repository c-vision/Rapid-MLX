# SPDX-License-Identifier: Apache-2.0
"""DS4 Downloader - handles model downloads with resume support and parallel chunks."""

from __future__ import annotations

import hashlib
import math
import time
from pathlib import Path
from typing import Optional
import urllib.parse

import requests

from vllm_mlx.tools.ds4_manager import DS4StatusManager


class DS4DownloadError(Exception):
    """Exception raised when DS4 download fails."""
    pass


class DS4Downloader:
    """Handles downloading models with resume support, parallel chunks, and integrity verification."""

    # Configuration constants
    CHUNK_SIZE =  SIZE = 5 * 1024 * 1024  # 5MB chunks for good throughput
    MAX_CONCURRENT_DOWNLOADS = 4
    REQUEST_TIMEOUT = 30
    CONNECT_TIMEOUT = 10
    RETRY_ATTEMPTS = 3
    RETRY_BACKOFF = 2.0

    def __init__(self, cache_dir: Optional[Path | str] = None):
        """
        Initialize the DS4 downloader.

        Args:
            cache_dir: Directory to store downloaded models (default: from env or default)
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "rapid-mlx" / "ds4"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Session for connection pooling and headers
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Rapid-MLX-DS4-Downloader/1.0",
            "Accept": "*/*",
        })

        # Status manager for tracking
        self.status_manager = DS4StatusManager(self.cache_dir)

    def _get_hf_download_url(self, model_id: str) -> Optional[str]:
        """
        Get HuggingFace download URL for a model ID.

        Returns the URL to download the model file, or None if not found.
        """
        try:
            # Clean model ID - remove any local path references
            if "/" in model_id:
                # HF format: org/model
                org_model = model_id
            else:
                # Local path - can't download from HF
                return None

            # Try common file extensions in order of preference
            extensions = [".bin", ".safetensors", ".pth", ".ckpt"]

            for ext in extensions:
                url = f"https://huggingface.co/{org_model}/resolve/main/{org_model.split('/')[-1]}{ext}"

                # Check if URL exists
                try:
                    response = self.session.head(url, timeout=self.CONNECT_TIMEOUT)
                    if response.status_code == 200:
                        return url
                except Exception:
                    continue

            # Try without extension (sometimes HF uses the org/model as filename)
            url = f"https://huggingface.co/{org_model}/resolve/main/{org_model}"
            try:
                response = self.session.head(url, timeout=self.CONNECT_TIMEOUT)
                if response.status_code == 200:
                    return url
            except Exception:
                pass

            # Last resort: try the model ID as-is
            url = f"https://huggingface.co/{org_model}/resolve/main/{org_model.split('/')[-1]}"
            try:
                response = self.session.head(url, timeout=self.CONNECT_TIMEOUT)
                if response.status_code == 200:
                    return url
            except Exception:
                pass

        except Exception:
            pass

        return None

    def _get_file_size(self, url: str) -> Optional[int]:
        """Get file size from HEAD request."""
        try:
            response = self.session.head(url, timeout=self.CONNECT_TIMEOUT)
            if response.status_code == 200:
                content_length = response.headers.get("Content-Length")
                if content_length:
                    return int(content_length)
        except Exception:
            pass
        return None

    def _calculate_chunk_ranges(self, total_size: int, num_chunks: int) -> list[tuple[int, int]]:
        """Calculate byte ranges for parallel chunks."""
        if total_size_chunks <= 0 or num_chunks <= 0:
            return []

        chunk_size = total_size // num_chunks
        remainder = total_size % num_chunks

        ranges = []
        start = 0
        for i in range(num_chunks):
            # Distribute remainder across first chunks
            current_chunk_size = chunk_size + (1 if i < remainder else 0)
            end = start + current_chunk_size
            ranges.append((start, end))
            start = end

        return ranges

    def _download_chunk(
        self,
        url: str,
        start: int,
        end: int,
        part_file: Path,
        retries: int = 0
    ) -> bool:
        """
        Download a single byte range chunk.

        Returns True on success, False on failure.
        """
        headers = {
            "Range": f"bytes={start}-{end - 1}" if end > start else "bytes=*/",
            "User-Agent": "Rapid-MLX-DS4-Downloader/1.0",
        }

        for attempt in range(retries + 1):
            try:
                response = self.session.get(
                    url,
                    headers=headers,
                    timeout=self.REQUEST_TIMEOUT,
                    stream=True
                )

                if response.status_code in (200, 206):  # OK or Partial Content
                    with open(part_file, "ab") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:  # filter out keep-alive chunks
                                f.write(chunk)
                    return True
                elif response.status_code == 416:  # Range not satisfiable
                    # This can happen if we're trying to download beyond file size
                    return True
                else:
                    if attempt == retries:
                        print(f"Chunk download failed with status {response.status_code}: {url}")
                        return False

            except requests.RequestException as e:
                if attempt == retries:
                    print(f"Chunk download error (attempt {attempt + 1}): {e}")
                    return False
                # Wait before retry
                time.sleep(self.RETRY_BACKOFF ** attempt)

        return False

    def _merge_chunks(self, model_name: str, num_chunks: int) -> Optional[Path]:
        """Merge all chunks into final file."""
        part_files = [self.cache_dir / f"{model_name}.part{i}" for i in range(num_chunks)]
        final_file = self.cache_dir / f"{model_name}"

        # Verify all part files exist
        for part_file in part_files:
            if not part_file.exists():
                print(f"Missing chunk file: {part_file}")
                return None

        try:
            with open(final_file, "wb") as output_file:
                for part_file in part_files:
                    with open(part_file, "rb") as input_file:
                        output_file.write(input_file.read())
                    # Remove chunk file after successful merge
                    part_file.unlink()

            return final_file
        except Exception as e:
            print(f"Failed to merge chunks: {e}")
            return None

    def _verify_file_integrity(self, filepath: Path, expected_size: Optional[int] = None) -> bool:
        """Verify downloaded file integrity."""
        if not filepath.exists():
            return False

        try:
            file_size = filepath.stat().st_size
            if file_size == 0:
                return False

            # Check file size if provided
            if expected_size is not None and file_size != expected_size:
                print(f"Size mismatch: expected {expected_size}, got {file_size}")
                return False

            # Basic integrity - file exists and has reasonable size
            return file_size > 1024  # At least 1KB

        except Exception:
            return False

    def download_model(self, model_id: str, show_progress: bool = True) -> tuple[bool, Optional[str]]:
        """
        Download a model with resume support and progress tracking.

        Returns:
            Tuple of (success: bool, file_path: Optional[str])
        """
        # Get download URL
        url = self._get_hf_download_url(model_id)
        if not url:
            print(f"Error: Could not determine download URL for model '{model_id}'")
            return False, None

        # Get file info
        total_size = self._get_file_size(url)
        if total_size is None:
            print(f"Warning: Could not determine file size for '{model_id}'. Proceeding without progress bar.")

        # Check current status
        status = self.status_manager.get_model_status(model_id)
        start_byte = int(status.get("last_byte", 0))
        existing_path = status.get("path")

        # If already completed, return early
        if status.get("status") == "completed" and existing_path and Path(existing_path).exists():
            if show_progress:
                print(f"✓ Model '{model_id}' already downloaded: {existing_path}")
            return True, existing_path

        # Prepare for download
        filename = model_id.split("/")[-1] if "/" in model_id else model_id
        part_file = self.cache_dir / f"{filename}.part"
        final_file = self.cache_dir / filename

        # Determine starting point
        if part_file.exists():
            start_byte = part_file.stat().st_size
            if show_progress:
                print(f"Resuming download of '{model_id}' from byte {start_byte:,}")
        elif final_file.exists():
            # Final file exists but status not completed - check if complete
            if self._verify_file_integrity(final_file, total_size):
                self.status_manager.set_model_status(
                    model_id,
                    "completed",
                    str(final_file),
                    last_byte=final_file.stat().st_size
                )
                if show_progress:
                    print(f"✓ Model '{model_id}' verified as complete: {final_file}")
                return True, str(final_file)
            else:
                # Corrupted file - remove and restart
                print(f"Corrupted file detected for '{model_id}', removing and restarting")
                final_file.unlink(missing_ok=True)
                start_byte = 0

        # Update status to downloading
        self.status_manager.set_model_status(
            model_id,
            "downloading",
            None,
            last_byte=start_byte
        )

        # Setup progress tracking
        if show_progress and total_size:
            print(f"\nDownloading '{model_id}' ({total_size / (1024*1024):.1f} MB)")
            start_time = time.time()

            def update_progress(downloaded: int):
                if total_size > 0:
                    percent = (downloaded / total_size) * 100
                    elapsed = time.time() - start_time
                    speed = downloaded / elapsed if elapsed > 0 else 0
                    eta = (total_size - downloaded) / speed if speed > 0 else 0
                    print(
                        f"\rProgress: {percent:.1f}% ({downloaded / (1024*1024):.1f}/{total_size / (1024*1024):.1f} MB) "
                        f"Speed: {speed / (1024*1024):.1f} MB/s ETA: {eta:.0f}s",
                        end="", flush=True
                    )
        else:
            def update_progress(downloaded: int):
                pass  # No-op if not showing progress

        # Download the remaining bytes
        success = False
        try:
            if total_size is not None and start_byte < total_size:
                # Download in chunks to show progress
                chunk_size = min(self.CHUNK_SIZE, max(8192, total_size // 100))  # At least 100 progress updates

                with open(part_file, "ab") as f:
                    downloaded = start_byte
                    while downloaded < total_size:
                        # Calculate chunk boundaries
                        chunk_end = min(downloaded + chunk_size, total_size)

                        # Download this chunk
                        headers = {
                            "Range": f"bytes={downloaded}-{chunk_end - 1}",
                            "User-Agent": "Rapid-MLX-DS4-Downloader/1.0",
                        }

                        for attempt in range(self.RETRY_ATTEMPTS + 1):
                            try:
                                response = self.session.get(
                                    url,
                                    headers=headers,
                                    timeout=self.REQUEST_TIMEOUT,
                                    stream=True
                                )

                                if response.status_code in (200, 206):
                                    for chunk in response.iter_content(chunk_size=8192):
                                        if chunk:
                                            f.write(chunk)
                                            downloaded += len(chunk)
                                            update_progress(downloaded)
                                    break  # Success, exit retry loop
                                elif attempt == self.RETRY_ATTEMPTS:
                                    print(f"\nFailed to download chunk after {self.RETRY_ATTEMPTS + 1} attempts")
                                    return False, None
                                else:
                                    time.sleep(self.RETRY_BACKOFF ** attempt)

                            except requests.RequestException as e:
                                if attempt == self.RETRY_ATTEMPTS:
                                    print(f"\nNetwork error downloading chunk: {e}")
                                    return False, None
                                time.sleep(self.RETRY_BACKOFF ** attempt)

                # Verify final file
                if self._verify_file_integrity(part_file, total_size):
                    # Rename part file to final
                    part_file.rename(final_file)
                    success = True
                else:
                    print(f"\nDownloaded file failed integrity check")
                    return False, None
            else:
                # No size info - download as single chunk
                headers = {
                    "Range": f"bytes={start_byte}-",
                    "User-Agent": "Rapid-MLX-DS4-Downloader/1.0",
                }

                response = self.session.get(
                    url,
                    headers=headers,
                    timeout=self.REQUEST_TIMEOUT,
                    stream=True
                )

                if response.status_code in (200, 206):
                    with open(part_file, "wb") as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                                update_progress(len(chunk))

                    if self._verify_file_integrity(part_file):
                        part_file.rename(final_file)
                        success = True
                    else:
                        print(f"\nDownloaded file failed integrity check")
                        return False, None
                else:
                    print(f"Failed to initiate download: HTTP {response.status_code}")
                    return False, None

        except Exception as e:
            print(f"\nUnexpected error during download: {e}")
            return False, None

        # Update final status
        if success and final_file.exists():
            file_size = final_file.stat().st_size
            self.status_manager.set_model_status(
                model_id,
                "completed",
                str(final_file),
                last_byte=file_size
            )
            if show_progress:
                print(f"\n✓ Successfully downloaded '{model_id}' to: {final_file}")
            return True, str(final_file)
        else:
            self.status_manager.set_model_status(
                model_id,
                "error",
                None,
                error="Download failed"
            )
            return False, None

# Convenience function for external use
def download_model(model_id: str, show_progress: bool = True) -> tuple[bool, Optional[str]]:
    """
    Convenience function to download a model using DS4.

    Args:
        model_id: Model identifier (e.g., "mlx-community/Qwen3.5-27B-8bit")
        show_progress: Whether to show progress bar

    Returns:
        Tuple of (success: bool, file_path: Optional[str])
    """
    downloader = DS4Downloader()
    return downloader.download_model(model_id, show_progress)