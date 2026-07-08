<p align="center">
<h1 align="center">Rapid-MLX NextGen</h1>

<p align="center">
  <strong>A community fork of Rapid-MLX — the fast local LLM inference engine for Apple Silicon.</strong>
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-blue.svg" alt="License"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.10+-blue.svg" alt="Python 3.10+"></a>
  <a href="https://support.apple.com/en-us/HT211814"><img src="https://img.shields.io/badge/Apple_Silicon-M1%20|%20M2%20|%20M3%20|%20M4-black.svg?logo=apple" alt="Apple Silicon"></a>
  <a href="https://github.com/raullenchai/Rapid-MLX"><img src="https://img.shields.io/badge/upstream-raullenchai%2FRapid--MLX-lightgrey.svg" alt="Upstream"></a>
</p>

## What is Rapid-MLX

Rapid-MLX is a local LLM inference engine for Apple Silicon Macs. It speaks the OpenAI Chat / Responses / Embeddings APIs, so anything that talks to ChatGPT — Cursor, Claude Code, Codex CLI, Aider, OpenCode, Qwen Code, Kilo Code, LangChain, PydanticAI, your own scripts — just works against a local `http://localhost:8000` endpoint. No cloud, no API key, no per-token cost. It's built on [MLX](https://github.com/ml-explore/mlx), Apple's array framework for Apple Silicon, and is tuned to be significantly faster than alternatives like Ollama or `mlx-lm serve` on the same hardware.

## What this fork adds

This is a personal fork tracking [raullenchai/Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) upstream, with the following additions layered on top:

### 1. User-configurable model aliases

A short name → full model mapping you control yourself, instead of only relying on the built-in curated alias list.

- Defined in `~/.rapid-mlx/models.yaml`, resolved with **higher priority** than the built-in aliases (`vllm_mlx/model_aliases.py`) but falling back to them transparently when a name isn't in your own config
- Validated on load — a malformed entry produces a clear error message instead of a silent misconfiguration
- Supports the same knobs the built-in aliases support (`model`, `quant`, `modality`, `hardware`, `min_vram_gb`, extra `args`), plus:

```yaml
aliases:
  fast:
    model: mlx-community/Qwen3.5-4B-4bit
    quant: 4bit
  local:
    model: mlx-community/Qwen3.5-Coder-27B-8bit
    path: /Volumes/External/models/qwen3.5-coder-27b   # see below
```

### 2. Model path overrides

Any alias (yours or built-in) can pin a `path:` field pointing at an already-downloaded model directory on disk — useful for models stored outside the default HuggingFace cache (external drives, shared network storage, air-gapped machines). Wired all the way through the CLI, so `rapid-mlx serve local` / `rapid-mlx chat local` just use the local copy without touching the network.

Full reference: [`vllm_mlx/alias_resolver.py`](vllm_mlx/alias_resolver.py), tests in [`tests/test_alias_resolver.py`](tests/test_alias_resolver.py).

### 3. Model download tooling

A parallel-download manager (`vllm_mlx/tools/model_downloader.py` + `model_download_status.py`) that fetches a full HuggingFace model repo (config, tokenizer, weight shards — not just one file) into a target directory, with resume support and SHA256 verification of LFS-tracked files. File transfer itself is delegated to `huggingface_hub.snapshot_download` (already a core dependency) rather than reimplementing HTTP range-request resume logic — `snapshot_download` already does that correctly, including per-file parallelism (default 8 workers, configurable — `--workers N` on `rapid-mlx pull --dest`, or `max_workers=` on `download_model()`).

Wired into the CLI as `rapid-mlx pull <repo> --dest <dir>` — with `--dest`, `pull` uses this downloader (resume + SHA256 verification, lands in `<dir>/<repo-name>`) instead of the default HuggingFace-cache path.

**Full example — download into a folder of your choice, then register it under an alias of your choice:**

```bash
# 1. Download a model. DIR can be anywhere — `~` is expanded, and the
#    model lands in DIR/<repo-name> (here: ~/my-models/SmolLM2-135M-Instruct-8bit).
rapid-mlx pull mlx-community/SmolLM2-135M-Instruct-8bit --dest ~/my-models
```

```yaml
# 2. Register that path under whatever short name you want in
#    ~/.rapid-mlx/models.yaml (created if it doesn't exist yet):
aliases:
  smol:
    model: mlx-community/SmolLM2-135M-Instruct-8bit
    path: ~/my-models/SmolLM2-135M-Instruct-8bit
```

```bash
# 3. Use your alias like any built-in one — no network access needed,
#    it serves straight from the path above. --served-model-name makes
#    the API expose it as "smol" too (otherwise it defaults to the
#    resolved path/repo id):
rapid-mlx serve smol --served-model-name smol
```

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"smol","messages":[{"role":"user","content":"Say hi in one short sentence."}]}'
```

Interrupted downloads resume correctly: `huggingface_hub` tracks partial files under `<dest_dir>/<repo-name>/.cache/huggingface/download/*.incomplete`, and re-running the same `rapid-mlx pull ... --dest ...` call skips whatever's already complete and only re-fetches what's missing — verified twice: once on a small test model, and again on a real 18 GB model (`mlx-community/gemma-4-31b-it-4bit`) interrupted mid-transfer by hand and re-run, which picked up exactly where it left off (already-complete files untouched, only the still-partial shards resumed) rather than starting over.

**Stalled-transfer watchdog.** `requests` (what `snapshot_download` uses under the hood) has no default read timeout — if a CDN connection goes quiet without a clean close (observed in the wild: stuck in `CLOSE_WAIT`, no error raised), the download just hangs forever with nothing to notice. `download_model()` now runs the transfer in a subprocess and watches the destination directory's total size: no growth for 5 minutes by default and it kills that subprocess, prints `Stalled: no progress for N min — restarting the transfer (attempt X/Y)`, and retries once by default (`STALL_RETRIES`, 2 attempts total per phase — a stall this deep is a strong enough signal that a third identical retry isn't worth the wait; see the Xet fallback below for the real remedy), relying on the same resume support so a restart continues from the stalled point instead of starting over. Both the timeout and the retry count are configurable — `rapid-mlx pull <repo> --dest <dir> --stall-timeout <minutes> --stall-retries <n>` (or `stall_minutes=`/`retries=` on `download_model()` directly). `rapid-mlx pull --dest` re-prompting on every resume was also fixed: the CLI's large-download confirmation gate only checked the default HuggingFace cache, so it never recognized a `--dest` folder that already had a download in progress — it now checks that folder directly and skips the prompt if anything's already there.

**Xet transfer failures — now handled automatically.** The watchdog above handles a connection going silent mid-transfer — it doesn't help when the transfer never delivers any bytes at all, which is a distinct failure mode of HuggingFace's Xet storage backend (many `mlx-community` repos use it instead of plain LFS). Symptom: the progress bar sits at `0%` / `0.00/<size>` for the entire stall window, identically on every retry, even though `huggingface.co` itself is reachable and responding fine — because Xet transfers go through a separate CAS/S3-style endpoint that can be blocked or broken independently of the main site. Retrying against the same transport doesn't help since the failure isn't transient.

If every attempt in a phase stalls in a row (as opposed to a real error like a bad token, which is never retried this way), that's taken as a strong signal the Xet endpoint itself is the problem — the downloader prints a heads-up and automatically switches to the classic HTTP/LFS path (`HF_HUB_DISABLE_XET=1`) for one more fresh set of attempts (same `--stall-retries` budget) before giving up. No manual intervention needed anymore; you can still set `HF_HUB_DISABLE_XET=1` yourself ahead of time to skip straight to the classic path.

Verified directly: `mlx-community/Mixtral-8x22B-4bit` (73.9 GiB) stalled at 0 bytes 3/3 times with Xet enabled (3x 5-minute timeouts, watchdog behaving correctly by giving up rather than hanging forever), then transferred at real throughput immediately once Xet was disabled.

**Authentication.** If `HF_TOKEN` is set in the environment, it's picked up explicitly and passed straight to `snapshot_download` — not just left to whatever implicit fallback `huggingface_hub` might do — and confirmed on the console (`Using HF_TOKEN from environment (...last 4 chars)`, never the full token) so there's no guessing whether it's actually being used. Without one, downloads print a heads-up that they're unauthenticated (HuggingFace's own stricter, slower rate limits apply) rather than silently taking longer with no explanation.

**Progress heartbeat.** `snapshot_download` goes quiet for stretches with no progress signal of its own — not just while resuming a partial download, but observed even on a completely fresh one (a small test model sat at the same byte count for 2.5 minutes, then finished in seconds — looks like HuggingFace-side throttling/backoff, not something under our control). On every poll (every `STALL_POLL_SECONDS`, 2s by default — cheap since it's just a recursive directory `stat`, not a network call) the watchdog prints its own status line — `... 8.7 GiB on disk (3.3 MiB/s), 450s elapsed` — regardless of whether anything actually stalled. The rate is the transfer speed since the *previous* poll, not just the cumulative total: a byte count that updates in place is easy to misread as frozen when it's really just slow (this came up directly — after a real Xet→HTTP fallback, one poll showed the same total as the last, which read as "did this die?" even though it hadn't); a moving (or honestly-zero) rate answers that at a glance. This line updates itself in place (carriage return, not a new line each time) instead of piling up dozens of near-identical lines over a long download — and `huggingface_hub`'s own progress bar, which used to race the heartbeat on the same terminal line and produce garbled output, is now suppressed (`HF_HUB_DISABLE_PROGRESS_BARS=1`) so there's a single coherent status line.

Verified end-to-end, literally running the three steps above: fresh download to a custom folder, kill-and-resume, register a custom alias with a path override, serve it, and get a real completion back through the API.

The same downloader is also available directly from Python for scripting: `from vllm_mlx.tools.model_downloader import download_model; download_model("org/repo", dest_dir="~/my-models")` — `rapid-mlx pull --dest` is a thin CLI wrapper around this same function. There's also an earlier, broken draft of the same idea kept around for reference (`model_download_draft.py`) — not used by anything, not exported, not maintained.

**Bugs found and fixed while verifying this end-to-end** (all pre-existing, not introduced by this session's work):
- `vllm_mlx/cli.py`'s alias-resolution step in `main()` unconditionally crashed with `NameError` — a call to an unimported `get_resolver()`, plus a dangling reference to an undefined `resolved` variable left over from merging the user-alias feature with upstream's own alias-resolution block. This affected **every** `rapid-mlx serve`/`chat`/`run` invocation, not just this download tooling.
- `model_downloader.py`'s "already downloaded" check was keyed only by model ID, not by destination — requesting the same model into a *different* folder than a previous run incorrectly reported the old location without downloading anything to the new one. Now the recorded path has to match the current call's target directory.
- `dest_dir="~/..."` wasn't tilde-expanded (`pathlib.Path` doesn't do that on its own), so a `~`-based destination would have tried to create a literal `~` directory. Fixed with `.expanduser()`.
- The user-alias config loader (`alias_resolver.py`) silently discarded **the entire `~/.rapid-mlx/models.yaml`** whenever any alias had a `hardware:` list (e.g. `hardware: [m1, m2, m3, m4]` — the format the config file's own template documents) — `_validate_hardware` tried to hash a list, raised `TypeError`, and the broad `except Exception: pass` around config loading ate it silently. This meant **Phase 1 (user-configurable aliases) never actually worked** for any config using the documented multi-hardware-tag format. Fixed to accept a list of tags, not just one.

### 4. Kept current with upstream

This fork is periodically synced with upstream's `main` branch (currently tracking v0.10.2) so it keeps upstream's bug fixes and new model support, on top of the additions above.

## Installation

This fork isn't published to PyPI or Homebrew — install it from source:

```bash
git clone https://github.com/c-vision/Rapid-MLX-NextGen.git
cd Rapid-MLX-NextGen
pip install -e ".[dev]"
```

Requires Python 3.10+ (macOS ships an older system Python — install a newer one first if needed, e.g. `brew install python@3.12`).

Try it:

```bash
rapid-mlx chat                      # chat with the default model in a REPL
rapid-mlx serve qwen3.5-4b-4bit     # or start an OpenAI-compatible HTTP server
```

**Multimodal models** (Qwen-VL, image input to Gemma 4, DiffusionGemma) need the vision extra: `pip install 'rapid-mlx[vision]'` (or `pip install -e '.[dev,vision]'` from source) — text-only chat/serve/tools work without it, but a vision route on a core-only install returns a bare 500 instead of an install hint.

For the full installation matrix (uv, one-line installer, Homebrew, pip) and everything else about running and configuring the engine, **see the upstream README** — this fork changes none of that behavior.

## Apple Silicon requirement and platform limitations

Rapid-MLX is built on [MLX](https://github.com/ml-explore/mlx), which only runs on **Apple Silicon** (M1 through M4 and later) Macs with Metal GPU support.

- **Not supported**: Intel Macs, Linux, Windows, or any non-Apple-Silicon hardware — MLX itself has no backend for them
- **Memory-bound**: model size you can run is limited by your Mac's unified memory (see the sizing table in the upstream README for what fits at each memory tier)
- This fork inherits the exact same platform constraints as upstream — nothing here changes the hardware story

## Original project

This is a fork, not a competing project — most of the engine, the full feature set, and all the heavy lifting come from the original:

**👉 [raullenchai/Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) — read the original README for the complete feature list, benchmarks, and configuration reference.**

## License

[Apache License 2.0](LICENSE)
