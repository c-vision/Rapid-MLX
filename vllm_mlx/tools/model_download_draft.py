# SPDX-License-Identifier: Apache-2.0
"""
Model download tool for Rapid-MLX (early draft, superseded by model_downloader.py)

This is an earlier, incomplete iteration of the download tooling — it has
several undefined-name bugs (see download_model/download_range below) and
was never wired into the CLI. model_downloader.py is the version that
actually runs. Kept here for reference only; not imported anywhere.

This module was meant to handle downloading models to the local cache
directory with:
- Parallel multi-part downloads
- Progress tracking
- Resume support
- SHA256 integrity verification
- Integration with Rapid-MLX alias system
"""

import os
import hashlib
import tempfile
import urllib.parse
from pathlib import Path
from typing import Optional, Dict, Any, List
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
import tqdm

from vllm_mlx.config import DEFAULT_CACHE_DIR
from vllm_mlx.utils import get_cache_dir, get_env_variable

# Constants
CHUNK_SIZE = 1024 * 64  # 64KB
MAX_RETRIES = 3
DOWNLOAD_DIR_NAME = "downloads"
DOWNLOAD_STATUS_FILE = "download_status.json"

class ModelDownloader:
    """Handles downloading models with resume support and integrity verification."""

    def __init__(self, cache_dir: str | None = None):
        """
        Initialize the model downloader.

        Args:
            cache_dir: Directory to store downloaded models (default: from env or default)
        """
        self.cache_dir = get_cache_dir() if cache_dir is None else cache_dir
        self.status_file = Path(self.cache_dir) / DOWNLOAD_STATUS_FILE
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _get_chunk_size(self, file_size: int) -> int:
        """Determine optimal chunk size based on file size."""
        if file_size < 10 * 1024**2:  # <10MB
            return 1024  # 1KB chunks for small files
        elif file_size < 100 * 1024**2:  # <100MB
            return 1024 * 8  # 8KB chunks for medium files
        return CHUNK_SIZE  # 64KB chunks for large files

    def _calculate_md5(self, data: bytes) -> str:
        """Calculate MD5 hash of data."""
        import hashlib
        return hashlib.md5(data).hexdigest()

    def _calculate_sha256(self, data: bytes) -> str:
        """Calculate SHA256 hash of data."""
        import hashlib
        return hashlib.sha256(data).hexdigest()

    def download_range(self, url: str, start: int, end: int, retries: int = MAX_RETRIES) -> bytes:
        """
        Download a byte range from a URL.

        Args:
            url: The URL to download from
            start: Start byte position
            end: End byte position
            retries: Number of retry attempts

        Returns:
            Downloaded bytes in the specified range
        """
        headers = {"Range": f"bytes={start}-{end-1}"}
        for attempt in range(retries):
            try:
                response = requests.get(url, headers=headers, stream=True, timeout=15)
                response.raise_for_status()

                content_length = response.headers.get("Content-Length", 0)
                if content_length is None:
                    content_length = 0

                if "Content-Range" in response.headers:
                    range_header = response.headers["Content-Range"]
                    match = response.headers["Content-Range"].split("/")
                    if len(match) > 1:
                        total_size = int(match[1])
                    else:
                        total_size = int(response.headers["Content-Length"])
                else:
                    total_size = int(response.headers.get("Content-Length", 0))

                # Check if range is valid
                if start >= total_length:
                    return b""

                content = response.content
                if len(content) > 0:
                    return content
            except Exception as e:
                if attempt == retries - 1:
                    raise Exception(f"Download failed after {retries} retries: {str(e)}")
                continue
        return b""

    def download_chunk(self, url: str, start: int, end: int, retries: int = MAX_RETRIES) -> bytes:
        """Download a single chunk with retries."""
        headers = {"Range": f"bytes={start}-{end-1}"}
        for attempt in range(retries):
            try:
                response = requests.get(url, headers=headers, timeout=15)
                response.raise_for_status()
                if response.status_code == 416:  # Range not satisfiable
                    return b""
                return response.content
            except Exception as e:
                if attempt == retries - 1:
                    raise Exception(f"Chunk download failed: {str(e)}")
        return b""

    def download_model(self, model_id: str, dest_path: str, show_progress: bool = True) -> None:
        """
        Download a model with parallel chunks.

        Args:
            model_id: HF model ID or name
            dest_path: Path to save the model
            show_progress: Show progress bar
        """
        # Extract filename from URL
        filename = model_id.split("/")[-1]
        if "." not in filename:
            filename += ".bin"  # Default extension

        dest_file = Path(dest_path) / filename
        dest_file.parent.mkdir(parents=True, exist_ok=True)

        # Get file size from HEAD request
        try:
            head_response = requests.head(url, timeout=10)
            total_size = int(response.headers.get("Content-Length", 0))
        except Exception:
            total_size = 0

        # Setup progress tracking
        if show_progress:
            progress = tqdm.tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {model_id}")

        # Create temp file
        temp_file = Path(temp_file.name)
        temp_file.touch()

        # Start parallel download
        threads = []
        chunk_offsets = []
        if total_size > 0:
            chunk_size = self._get_chunk_size(total_size)
            num_chunks = max(1, total_size // chunk_size)
            for i in range(0, total_size, chunk_size):
                start = i
                end = min(i + chunk_size, total_size)
                chunks.append((start, end))
                threads.append(Thread(target=self._download_chunk, args=(url, start, end)))
        else:
            # Unknown size - read sequentially
            range_header = response.headers.get("Content-Range")
            if range_header:
                total_size = int(range_header.split("/")[-1])
                chunks = [(0, total_size)]
                threads.append(Thread(target=self._download_chunk, args=(url, 0, total_size)))
            else:
                # Stream download without known size
                chunks = [(0, 0)]  # Single chunk
                threads.append(Thread(target=self._download_chunk, args=(url, 0, 0)))

        # Start threads
        results = []
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        # Write chunks to file in order
        with open(temp_file, 'wb') as f:
            for i, chunk_data in enumerate(results):
                f.write(chunk_data)

        # Move temp file to final location
        temp_file.rename(Path(dest_file))

        if show_progress:
            progress.close()

    def download_all_chunks(self, url: str, dest_path: str, chunk_size: int = CHUNK_SIZE) -> List[bytes]:
        """Internal method to download all chunks from a URL."""
        # Get file size
        try:
            head_response = requests.head(url, timeout=10)
            total_size = int(head_response.headers.get("Content-Length", 0))
        except Exception:
            total_size = 0

        # Generate chunk offsets
        chunk_offsets = []
        offset = 0
        while offset < total_size or total_size == 0:
            start = offset
            end = offset + chunk_size if offset + chunk_size <= total_size else total_size
            chunk_offsets.append((start, end if offset + chunk_size < total_size else 0))
            offset += chunk_size if offset + chunk_size <= total_size else 0

        return chunk_offsets

    def _download_chunk(self, url: str, start: int, end: int) -> bytes:
        """Download a specific byte range."""
        headers = {"Range": f"bytes={start}-{end-1}" if end > 0 else "bytes=*/" + str(end)}
        response = requests.get(url, headers=headers, stream=True, timeout=15)
        response.raise_for_status()
        return response.content