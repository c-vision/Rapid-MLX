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
import multiprocessing
import os
import time
from pathlib import Path
from typing import Optional

import requests
from huggingface_hub import snapshot_download

from vllm_mlx.tools.model_download_status import ModelDownloadStatusManager


class ModelDownloadError(Exception):
    """Exception raised when a model download fails."""
    pass


def _masked_token(token: str) -> str:
    """Last 4 chars only — enough to confirm "yes, this is my token" without
    printing the whole secret to a terminal or log file."""
    return f"...{token[-4:]}" if len(token) > 4 else "...(short token)"


def _snapshot_download_worker(
    model_id: str, final_dir: str, max_workers: int, token: Optional[str], result_queue
) -> None:
    """Runs snapshot_download in its own process so a stalled transfer can
    be killed outright. ``requests`` (used under the hood) has no default
    read timeout — a CDN connection that goes silent without a clean close
    leaves the process blocked in a socket read forever, with nothing to
    detect "no bytes for N minutes" from inside that same call. A
    subprocess can be terminated from outside; a stuck thread in this
    process couldn't be.
    """
    try:
        snapshot_download(repo_id=model_id, local_dir=final_dir, max_workers=max_workers, token=token)
        result_queue.put(("ok", None))
    except Exception as e:
        result_queue.put(("error", str(e)))


def _dir_size_bytes(path: Path) -> int:
    """Total size of all files under path, 0 if it doesn't exist yet."""
    if not path.exists():
        return 0
    total = 0
    for f in path.rglob("*"):
        try:
            if f.is_file():
                total += f.stat().st_size
        except OSError:
            continue
    return total


def _format_size(num_bytes: float) -> str:
    for unit in ("B", "KiB", "MiB", "GiB"):
        if num_bytes < 1024:
            return f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} TiB"


def _env_is_true(name: str) -> bool:
    return os.environ.get(name, "").strip().upper() in {"1", "TRUE", "YES", "ON"}


def _print_status_line(text: str, last_len: int) -> int:
    """Overwrites the current terminal line instead of stacking a new one
    each call — used for the heartbeat, which would otherwise print dozens
    of near-identical lines over a long download."""
    pad = max(0, last_len - len(text))
    print("\r" + text + " " * pad, end="", flush=True)
    return len(text)


def _end_status_line(last_len: int) -> None:
    """Moves past an in-place status line so the next print() starts on
    its own fresh line instead of overwriting/appending to it."""
    if last_len:
        print()


class ModelDownloader:
    """Downloads full HuggingFace model repos with resume support and integrity verification."""

    MAX_CONCURRENT_DOWNLOADS = 8
    CONNECT_TIMEOUT = 10
    STALL_MINUTES = 5
    STALL_RETRIES = 1
    STALL_POLL_SECONDS = 2
    XET_DISABLE_VAR = "HF_HUB_DISABLE_XET"

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

    def _run_stall_attempts(
        self,
        model_id: str,
        final_dir: Path,
        show_progress: bool,
        stall_seconds: float,
        stall_minutes: float,
        token: Optional[str],
        max_attempts: int,
        max_workers: int,
    ) -> tuple[bool, Optional[str], bool]:
        """Runs up to max_attempts download attempts, restarting the
        subprocess whenever final_dir's total size hasn't grown in
        stall_seconds (a CDN connection stuck in CLOSE_WAIT that `requests`
        never notices, since it has no default read timeout — the process
        would otherwise block in a socket read forever).

Prints a heartbeat on every poll (every STALL_POLL_SECONDS) regardless
        of whether anything stalled — snapshot_download goes quiet for a
        while up front verifying already-downloaded partial files before
        resuming, and that quiet phase has no progress signal of its own.
        Includes the transfer rate since the last poll, not just the
        cumulative total — a single in-place-updating byte count is easy
        to mistake for frozen output; a moving rate is the difference
        between "is this actually doing anything?" and "it says 8.7 GiB,
        same as it said a moment ago, is that stuck or just slow?". Both
        the heartbeat and the huggingface_hub tqdm bar it would otherwise
        race against on the same terminal line are handled by the caller.

        Returns (ok, err, all_attempts_stalled) — the third value is True
        only when every attempt stalled out (as opposed to a real error
        like a bad token), since that's the only case where switching
        transport (Xet on/off) has any chance of helping.
        """
        ctx = multiprocessing.get_context("spawn")

        for attempt in range(1, max_attempts + 1):
            result_queue = ctx.Queue()
            proc = ctx.Process(
                target=_snapshot_download_worker,
                args=(model_id, str(final_dir), max_workers, token, result_queue),
            )
            proc.start()

            start_time = time.monotonic()
            last_size = _dir_size_bytes(final_dir)
            last_progress = start_time
            poll_size = last_size
            poll_time = start_time
            last_line_len = 0
            stalled = False

            while proc.is_alive():
                time.sleep(self.STALL_POLL_SECONDS)
                size = _dir_size_bytes(final_dir)
                now = time.monotonic()
                if size > last_size:
                    last_size = size
                    last_progress = now
                elif now - last_progress > stall_seconds:
                    stalled = True
                    if show_progress:
                        _end_status_line(last_line_len)
                        print(
                            f"  Stalled: no progress for {stall_minutes:g} min — "
                            f"restarting the transfer (attempt {attempt}/{max_attempts})..."
                        )
                    proc.terminate()
                    proc.join(timeout=10)
                    if proc.is_alive():
                        proc.kill()
                        proc.join()
                    break

                if show_progress:
                    interval = now - poll_time
                    rate = (size - poll_size) / interval if interval > 0 else 0.0
                    last_line_len = _print_status_line(
                        f"  ... {_format_size(size)} on disk ({_format_size(rate)}/s), "
                        f"{now - start_time:.0f}s elapsed",
                        last_line_len,
                    )
                poll_size = size
                poll_time = now

            if not stalled:
                if show_progress:
                    _end_status_line(last_line_len)
                proc.join()
                if result_queue.empty():
                    return False, "download process exited without reporting a result", False
                status, err = result_queue.get()
                return (True, None, False) if status == "ok" else (False, err, False)
            # Stalled: loop retries, snapshot_download resumes the partial
            # files already on disk instead of starting over.

        return False, f"gave up after {max_attempts} stalled retries", True

    def _download_with_stall_watchdog(
        self,
        model_id: str,
        final_dir: Path,
        show_progress: bool = True,
        stall_minutes: Optional[float] = None,
        token: Optional[str] = None,
        retries: Optional[int] = None,
        max_workers: Optional[int] = None,
    ) -> tuple[bool, Optional[str]]:
        """Runs the download, retrying stalls, and escalating once if every
        attempt in a row stalls out.

        Stalling on *every* attempt (as opposed to one transient blip) is
        the signature of HuggingFace's Xet transfer backend being broken
        or blocked — observed directly: huggingface_hub's own transfer
        progress (not just our polling) sits at 0 bytes for the entire
        stall window, identically on every retry, while huggingface.co
        itself is reachable fine. No amount of retrying the same transport
        fixes that. So once every attempt in a phase has stalled, if Xet
        isn't already disabled, switch to the classic HTTP/LFS path
        (HF_HUB_DISABLE_XET=1) and run one more fresh set of attempts
        before giving up for good.

        `retries` is retries *beyond* the first attempt (default
        STALL_RETRIES = 1, i.e. 2 attempts total per phase) — a stalled
        transfer this deep is already a strong signal something structural
        is wrong, not a blip a third or fourth identical retry would fix;
        the real remedy is the Xet fallback below, not more attempts on
        the same transport.
        """
        # huggingface_hub's own tqdm bar and our heartbeat below both write
        # to the same terminal line independently — left enabled, they
        # interleave into garbled output. Ours is the one meant to survive
        # subprocess restarts, so it's the one that stays.
        os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")

        stall_minutes = stall_minutes if stall_minutes is not None else self.STALL_MINUTES
        stall_seconds = stall_minutes * 60
        max_attempts = (retries if retries is not None else self.STALL_RETRIES) + 1
        max_workers = max_workers if max_workers is not None else self.MAX_CONCURRENT_DOWNLOADS

        ok, err, all_stalled = self._run_stall_attempts(
            model_id, final_dir, show_progress, stall_seconds, stall_minutes, token, max_attempts, max_workers
        )
        if ok or not all_stalled or _env_is_true(self.XET_DISABLE_VAR):
            return ok, err

        if show_progress:
            print(
                f"  Still stalled after {max_attempts} tries — this looks like HuggingFace's Xet "
                f"transfer backend, not a slow connection. Switching to the classic HTTP/LFS transfer "
                f"({self.XET_DISABLE_VAR}=1) and trying again..."
            )
        os.environ[self.XET_DISABLE_VAR] = "1"

        ok, err, _ = self._run_stall_attempts(
            model_id, final_dir, show_progress, stall_seconds, stall_minutes, token, max_attempts, max_workers
        )
        return ok, err

    def download_model(
        self,
        model_id: str,
        dest_dir: Optional[Path | str] = None,
        show_progress: bool = True,
        stall_minutes: Optional[float] = None,
        retries: Optional[int] = None,
        max_workers: Optional[int] = None,
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
            stall_minutes: Minutes without progress before the transfer is
                considered stalled and restarted. Defaults to STALL_MINUTES
                (5) if not given.
            retries: Retries beyond the first attempt before giving up on a
                phase (2 attempts total by default) — see STALL_RETRIES.
                Defaults to STALL_RETRIES (1) if not given.
            max_workers: Parallel per-file transfers passed straight to
                `snapshot_download`. Defaults to MAX_CONCURRENT_DOWNLOADS
                (8) if not given.

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

        token = os.environ.get("HF_TOKEN")
        if show_progress:
            if token:
                print(f"  Using HF_TOKEN from environment ({_masked_token(token)})")
            else:
                print("  No HF_TOKEN found in environment — downloading unauthenticated (slower, stricter rate limits).")

        self.status_manager.set_model_status(model_id, "downloading")
        if show_progress:
            print(f"Downloading '{model_id}' into {final_dir} ...")

        ok, err = self._download_with_stall_watchdog(
            model_id,
            final_dir,
            show_progress=show_progress,
            stall_minutes=stall_minutes,
            token=token,
            retries=retries,
            max_workers=max_workers,
        )
        if not ok:
            self.status_manager.set_model_status(model_id, "error", error=err or "download failed")
            if show_progress:
                print(f"✗ Download failed for '{model_id}': {err}")
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
    stall_minutes: Optional[float] = None,
    retries: Optional[int] = None,
    max_workers: Optional[int] = None,
) -> tuple[bool, Optional[str]]:
    """
    Convenience function to download a model.

    Args:
        model_id: Model identifier (e.g., "mlx-community/Qwen3.5-27B-4bit")
        dest_dir: Parent directory to download into (model lands in
            dest_dir/<repo-name>). Defaults to the downloader's cache dir.
        show_progress: Whether to show progress messages.
        stall_minutes: Minutes without progress before restarting the
            transfer. Defaults to ModelDownloader.STALL_MINUTES (5).
        retries: Retries beyond the first attempt before giving up on a
            phase. Defaults to ModelDownloader.STALL_RETRIES (1).
        max_workers: Parallel per-file transfers. Defaults to
            ModelDownloader.MAX_CONCURRENT_DOWNLOADS (8).

    Returns:
        Tuple of (success: bool, final directory path or None on failure)
    """
    downloader = ModelDownloader()
    return downloader.download_model(
        model_id,
        dest_dir=dest_dir,
        show_progress=show_progress,
        stall_minutes=stall_minutes,
        retries=retries,
        max_workers=max_workers,
    )
