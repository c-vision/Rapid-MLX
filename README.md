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

A download manager (`vllm_mlx/tools/model_downloader.py` + `model_download_status.py`) that fetches a full HuggingFace model repo (config, tokenizer, weight shards — not just one file) into a target directory, with resume support and SHA256 verification of LFS-tracked files. Transfer defaults to plain `curl` (see *Xet transfer failures* further down for why); `huggingface_hub.snapshot_download` is still available as an opt-in fallback (`--no-curl`).

Wired into the CLI as `rapid-mlx pull <repo> --dest <dir>` — with `--dest`, `pull` uses this downloader (resume + SHA256 verification, lands in `<dir>/<repo-name>`) instead of the default HuggingFace-cache path.

**`rapid-mlx pull <repo> --dest <dir>` — all parameters:**

| Flag | Default | What it does |
|---|---|---|
| `--dest DIR` | *(required for this downloader)* | Parent directory to download into — model lands in `DIR/<repo-name>`. Without `--dest`, `pull` uses the default HuggingFace-cache path instead (none of the flags below apply). |
| `--stall-timeout MINUTES` | `5` | Minutes without progress before a transfer is considered stalled and restarted (resumes, doesn't start over). |
| `--stall-retries N` | `1` | Retries beyond the first attempt before giving up on a phase (2 attempts total by default). With `--no-curl`, a stall this deep also triggers the automatic Xet→HTTP fallback (see below) before giving up. |
| `--no-curl` | off (curl is used) | Use `huggingface_hub` for the transfer instead of the default plain-curl downloader. See *Xet transfer failures* below for why curl is the default. `--workers` and `--disable-xet` only apply with `--no-curl`. |
| `--workers N` | `8` | *(only with `--no-curl`)* Parallel per-file transfers passed straight to `snapshot_download`. Curl mode is always sequential, one file at a time. |
| `--disable-xet` | off | *(only with `--no-curl`)* Skip HuggingFace's Xet transport and start straight on the classic HTTP/LFS path. Curl never uses Xet at all, so this has no effect without `--no-curl`. |
| `--no-token` | off (use `HF_TOKEN` if set) | Ignore `HF_TOKEN` even if it's present and download unauthenticated. See *Authentication* below. |

Everything below explains the *why* behind these defaults in more detail.

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

Interrupted downloads resume correctly regardless of transfer mode: curl (default) resumes each file with its own `-C -` flag, writing straight to the final filename — an interrupted file just picks up from its own last byte on retry, no separate staging area. `huggingface_hub` (`--no-curl`) tracks partial files under `<dest_dir>/<repo-name>/.cache/huggingface/download/*.incomplete` instead, and re-running the same `rapid-mlx pull ... --dest ...` call skips whatever's already complete and only re-fetches what's missing — verified twice: once on a small test model, and again on a real 18 GB model (`mlx-community/gemma-4-31b-it-4bit`) interrupted mid-transfer by hand and re-run, which picked up exactly where it left off (already-complete files untouched, only the still-partial shards resumed) rather than starting over.

**Why curl is the default.** Chasing down a real stuck transfer (`mlx-community/Mixtral-8x22B-4bit`, 73.9 GiB) led through several dead ends before finding the actual cause — each ruled out with a direct, controlled test rather than assumed: not the Xet backend (disabling it didn't help), not a stale `HF_TOKEN` (unauthenticated stalled too, and vice versa), not `multiprocessing.Queue` or spawn-vs-fork (isolated and ruled out), not a code regression from syncing upstream (pre-merge code reproduced the identical stall in a side-by-side worktree comparison). What finally isolated it: a single plain `curl --range` request to the exact same file, at the exact same moment `huggingface_hub` was sitting at 0 bytes, transferred real data immediately. `huggingface_hub` issues many more HTTP requests per download than curl does (metadata lookups, redirect resolution, per-file/per-worker parallelism) — consistent with HuggingFace throttling that higher-request-volume automated client pattern specifically, independent of the content or network itself. Since curl demonstrably keeps working in exactly the conditions where `huggingface_hub` didn't, it's now the default transfer mechanism (`_curl_download_worker`) — one file at a time, deliberately not parallelized, to stay close to the low-volume request pattern that was actually observed to work. `--no-curl` switches back to `huggingface_hub` (keeps Xet support and parallel per-file workers) if you'd rather have that trade-off.

**Stalled-transfer watchdog.** *(Applies to both transfer modes — curl and `--no-curl`.)* `requests` (what `snapshot_download` uses under the hood) has no default read timeout — if a CDN connection goes quiet without a clean close (observed in the wild: stuck in `CLOSE_WAIT`, no error raised), the download just hangs forever with nothing to notice. `download_model()` now runs the transfer in a subprocess and watches the destination directory's total size: no growth for 5 minutes by default and it kills that subprocess, prints `Stalled: no progress for N min — restarting the transfer (attempt X/Y)`, and retries once by default (`STALL_RETRIES`, 2 attempts total per phase — a stall this deep is a strong enough signal that a third identical retry isn't worth the wait; see the Xet fallback below for the real remedy), relying on the same resume support so a restart continues from the stalled point instead of starting over. Both the timeout and the retry count are configurable — `rapid-mlx pull <repo> --dest <dir> --stall-timeout <minutes> --stall-retries <n>` (or `stall_minutes=`/`retries=` on `download_model()` directly). `rapid-mlx pull --dest` re-prompting on every resume was also fixed: the CLI's large-download confirmation gate only checked the default HuggingFace cache, so it never recognized a `--dest` folder that already had a download in progress — it now checks that folder directly and skips the prompt if anything's already there.

**Xet transfer failures — now handled automatically.** *(Applies to `--no-curl` mode only — curl never touches the Xet backend, so this whole failure mode and its auto-fallback simply don't apply when using the default transfer.)* The watchdog above handles a connection going silent mid-transfer — it doesn't help when the transfer never delivers any bytes at all, which is a distinct failure mode of HuggingFace's Xet storage backend (many `mlx-community` repos use it instead of plain LFS). Symptom: the progress bar sits at `0%` / `0.00/<size>` for the entire stall window, identically on every retry, even though `huggingface.co` itself is reachable and responding fine — because Xet transfers go through a separate CAS/S3-style endpoint that can be blocked or broken independently of the main site. Retrying against the same transport doesn't help since the failure isn't transient.

If every attempt in a phase stalls in a row (as opposed to a real error like a bad token, which is never retried this way), that's taken as a strong signal the Xet endpoint itself is the problem — the downloader prints a heads-up and automatically switches to the classic HTTP/LFS path (`HF_HUB_DISABLE_XET=1`) for one more fresh set of attempts (same `--stall-retries` budget) before giving up. No manual intervention needed anymore.

If a repo is *already known* to stall on Xet (e.g. a previous run already had to escalate), waiting through that first phase again just to rediscover the same thing is wasted time — pass `--disable-xet` (or `disable_xet=True` on `download_model()`) to start straight on the classic path instead. Default is off, so the normal auto-escalating behavior above is unchanged unless you opt in.

Verified directly: `mlx-community/Mixtral-8x22B-4bit` (73.9 GiB) stalled at 0 bytes 3/3 times with Xet enabled (3x 5-minute timeouts, watchdog behaving correctly by giving up rather than hanging forever), then transferred at real throughput immediately once Xet was disabled. That same repo went on to stall completely under `huggingface_hub` even with Xet off — see *Why curl is the default* above — and completed the full 73.9 GiB with zero stalls once switched to curl, which is why curl is the default and this whole section is scoped to `--no-curl`.

**Authentication.** If `HF_TOKEN` is set in the environment, it's picked up explicitly and passed straight to `snapshot_download` — not just left to whatever implicit fallback `huggingface_hub` might do — and confirmed on the console (`Using HF_TOKEN from environment (...last 4 chars)`, never the full token) so there's no guessing whether it's actually being used. Without one, downloads print a heads-up that they're unauthenticated (HuggingFace's own stricter, slower rate limits apply) rather than silently taking longer with no explanation.

**`--no-token`.** HuggingFace tracks per-token and per-IP rate limits separately — a token that's made a lot of requests for a given repo (heavy retry traffic on a flaky transfer, for instance) can end up stalling completely while unauthenticated access to the very same repo works fine, since it falls under a different limit entirely. Confirmed directly: a `Mixtral-8x22B-4bit` pull stalled at 0 bytes for two full 5-minute windows with `HF_TOKEN` set, then completed normally once `--no-token` forced unauthenticated access. Pass `--no-token` (or `no_token=True` on `download_model()`) to ignore `HF_TOKEN` even if it's present — prints a heads-up naming the ignored token (masked) so it's clear what changed. Default is off (use `HF_TOKEN` when present, same as always); this trades away the higher unauthenticated rate limits, so it's a "try this if a transfer stalls repeatedly despite retries" knob, not a default.

**Progress heartbeat.** `snapshot_download` goes quiet for stretches with no progress signal of its own — not just while resuming a partial download, but observed even on a completely fresh one (a small test model sat at the same byte count for 2.5 minutes, then finished in seconds — looks like HuggingFace-side throttling/backoff, not something under our control). On every poll (every `STALL_POLL_SECONDS`, 2s by default — cheap since it's just a recursive directory `stat`, not a network call) the watchdog prints its own status line — `... 8.7 GiB / 42.8 GiB (20%) (3.3 MiB/s), 450s elapsed` — regardless of whether anything actually stalled. The total and percentage come from a best-effort HuggingFace metadata lookup (the same `estimate_repo_size_bytes` the CLI's own download-size confirmation prompt uses) and fall back to a bare on-disk byte count when that lookup fails (network down, gated repo, HF outage) rather than blocking or erroring. The rate is the transfer speed since the *previous* poll, not just the cumulative total: a byte count that updates in place is easy to misread as frozen when it's really just slow (this came up directly — after a real Xet→HTTP fallback, one poll showed the same total as the last, which read as "did this die?" even though it hadn't); a moving (or honestly-zero) rate answers that at a glance. This line updates itself in place (carriage return, not a new line each time) instead of piling up dozens of near-identical lines over a long download — and `huggingface_hub`'s own progress bar, which used to race the heartbeat on the same terminal line and produce garbled output, is now suppressed (`HF_HUB_DISABLE_PROGRESS_BARS=1`) so there's a single coherent status line.

Verified end-to-end, literally running the three steps above: fresh download to a custom folder, kill-and-resume, register a custom alias with a path override, serve it, and get a real completion back through the API.

**Real-world model catalog.** `rapid-mlx pull` commands for a working set of models actually used alongside this fork, grouped by family. Every repo listed here was verified live before being added (HuggingFace API returns the repo and it contains real `.safetensors` files, not an empty/placeholder repo).

#### DeepSeek

```bash
# DeepSeek-R1-Distill-Qwen-32B — 8-bit
rapid-mlx pull mlx-community/DeepSeek-R1-Distill-Qwen-32B-MLX-8Bit --dest ~/ai/Models --stall-timeout 5 --stall-retries 1
```

```bash
# DeepSeek-R1-Distill-Qwen-32B — 4-bit
rapid-mlx pull mlx-community/DeepSeek-R1-Distill-Qwen-32B-4bit --dest ~/ai/Models --stall-timeout 5 --stall-retries 1
```

```bash
# DeepSeek V4 Flash 158B-A13B — 4-bit
rapid-mlx pull mlx-community/deepseek-ai-DeepSeek-V4-Flash-4bit --dest ~/ai/Models --stall-timeout 5 --stall-retries 1
```

#### Qwen 3.5

```bash
# Qwen3.5-27B — 4-bit
rapid-mlx pull mlx-community/Qwen3.5-27B-4bit --dest ~/ai/Models --stall-timeout 5 --stall-retries 1
```

```bash
# Qwen3.5-27B Claude-4.6-Opus Distilled — 4-bit
rapid-mlx pull mlx-community/Qwen3.5-27B-Claude-4.6-Opus-Distilled-MLX-4bit --dest ~/ai/Models --stall-timeout 5 --stall-retries 1
```

#### Qwen 3.6

```bash
# Qwen3.5-122B-A10B — mxfp4 (MoE)
rapid-mlx pull mlx-community/Qwen3.5-122B-A10B-mxfp4 --dest ~/ai/Models --stall-timeout 5 --stall-retries 1
```

```bash
# Qwen3.6-35B-A3B — OptiQ 4-bit
rapid-mlx pull mlx-community/Qwen3.6-35B-A3B-OptiQ-4bit --dest ~/ai/Models --stall-timeout 5 --stall-retries 1
```

```bash
# Qwen3.6-40B Claude-like — 8-bit
rapid-mlx pull mlx-community/Qwen3.6-40B-Claude-4.6-Opus-Deckard-Heretic-Uncensored-Thinking-8bit --dest ~/ai/Models --stall-timeout 5 --stall-retries 1
```

#### Gemma 4

```bash
# Gemma-4-31B-it — 4-bit
rapid-mlx pull mlx-community/gemma-4-31b-it-4bit --dest ~/ai/Models --stall-timeout 5 --stall-retries 1
```

```bash
# Gemma-4-12B-coder — 4-bit
rapid-mlx pull mlx-community/gemma-4-12b-coder-fable5-composer2.5-4bit --dest ~/ai/Models --stall-timeout 5 --stall-retries 1
```

```bash
# Gemma-4-12B-coder — 8-bit
rapid-mlx pull mlx-community/gemma-4-12b-coder-fable5-composer2.5-8bit --dest ~/ai/Models --stall-timeout 5 --stall-retries 1
```

#### Devstral

```bash
rapid-mlx pull mlx-community/Devstral-Samll-2507-bf16 --dest ~/ai/Models --stall-timeout 5 --stall-retries 1
```

#### Mistral

```bash
# Mixtral-8x22B — 4-bit (known to stall on Xet — see above)
rapid-mlx pull mlx-community/Mixtral-8x22B-4bit --dest ~/ai/Models --stall-timeout 5 --stall-retries 1
```

```bash
rapid-mlx pull mlx-community/Mistral-Large-Instruct-2407-4bit --dest ~/ai/Models --stall-timeout 5 --stall-retries 1
```

#### Unlimited-OCR (MLX)

```bash
# Block-float MX FP8 (recommended for quality/performance)
rapid-mlx pull sahilchachra/unlimited-ocr-mxfp8-mlx --dest ~/ai/Models --stall-timeout 5 --stall-retries 1
```

#### Additional models

```bash
# Nemotron-3-Super-120B — 6-bit (~98GB)
rapid-mlx pull mlx-community/Nemotron-3-Super-120B-A12B-MLX-6bit --dest ~/ai/Models --stall-timeout 5 --stall-retries 1
```

```bash
# Qwen3-Coder-Next — 8-bit
rapid-mlx pull mlx-community/Qwen3-Coder-Next-8bit --dest ~/ai/Models --stall-timeout 5 --stall-retries 1
```

```bash
# Qwopus3.6-35B-A3B-Coder — 8-bit
rapid-mlx pull mlx-community/Qwopus3.6-35B-A3B-Coder-8bit --dest ~/ai/Models --stall-timeout 5 --stall-retries 1
```

```bash
# GLM-5.2 — 4-bit
rapid-mlx pull mlx-community/GLM-5.2-mxfp4 --dest ~/ai/Models --stall-timeout 5 --stall-retries 1
```

```bash
# Ornith-1.0-35B — 8-bit
rapid-mlx pull mlx-community/Ornith-1.0-35B-8bit --dest ~/ai/Models --stall-timeout 5 --stall-retries 1
```

```bash
# GPT-OSS-120B
rapid-mlx pull mlx-community/gpt-oss-120b-mxfp4-bf16 --dest ~/ai/Models --stall-timeout 5 --stall-retries 1
```

The same downloader is also available directly from Python for scripting: `from vllm_mlx.tools.model_downloader import download_model; download_model("org/repo", dest_dir="~/my-models")` — `rapid-mlx pull --dest` is a thin CLI wrapper around this same function. There's also an earlier, broken draft of the same idea kept around for reference (`model_download_draft.py`) — not used by anything, not exported, not maintained.

**Bugs found and fixed while verifying this end-to-end** (all pre-existing, not introduced by this session's work):
- `vllm_mlx/cli.py`'s alias-resolution step in `main()` unconditionally crashed with `NameError` — a call to an unimported `get_resolver()`, plus a dangling reference to an undefined `resolved` variable left over from merging the user-alias feature with upstream's own alias-resolution block. This affected **every** `rapid-mlx serve`/`chat`/`run` invocation, not just this download tooling.
- `model_downloader.py`'s "already downloaded" check was keyed only by model ID, not by destination — requesting the same model into a *different* folder than a previous run incorrectly reported the old location without downloading anything to the new one. Now the recorded path has to match the current call's target directory.
- `dest_dir="~/..."` wasn't tilde-expanded (`pathlib.Path` doesn't do that on its own), so a `~`-based destination would have tried to create a literal `~` directory. Fixed with `.expanduser()`.
- The user-alias config loader (`alias_resolver.py`) silently discarded **the entire `~/.rapid-mlx/models.yaml`** whenever any alias had a `hardware:` list (e.g. `hardware: [m1, m2, m3, m4]` — the format the config file's own template documents) — `_validate_hardware` tried to hash a list, raised `TypeError`, and the broad `except Exception: pass` around config loading ate it silently. This meant **Phase 1 (user-configurable aliases) never actually worked** for any config using the documented multi-hardware-tag format. Fixed to accept a list of tags, not just one.

### 4. Kept current with upstream

This fork is periodically synced with upstream's `main` branch (currently tracking v0.10.2) so it keeps upstream's bug fixes and new model support, on top of the additions above.

### 5. PFlash now respects `--no-mllm`/`--text-only`

`serve`/`bench`/the standalone server entrypoint all computed the `is_mllm` flag for PFlash's multimodal-rejection gate as `args.mllm or is_mllm_model(args.model)` — none of the three checked `args.no_mllm`, even though `--no-mllm`/`--text-only` forces `force_text=True` at the engine boundary, which sets `BatchedEngine._is_mllm = False` unconditionally (`engine/batched.py`) and routes the request through the exact same plain-text scheduler path any text-only checkpoint uses.

So a genuinely vision-capable checkpoint (real `vision_tower` weights in its safetensors — not the `#393` text-only-fork case `is_mllm_model()` already handles) forced into `--text-only` mode still had `--pflash` rejected for a code path it was never going to run. Verified directly against `mlx-community/Qwen3.6-27B-4bit`: 333 real `vision_tower.*` tensors in `model.safetensors.index.json`, confirming this isn't the `#393` case — yet `rapid-mlx serve <path> --text-only --no-mllm --pflash auto` failed immediately with `--pflash is not supported for multimodal models`.

Fixed by extracting the computation into `pflash.resolve_effective_is_mllm()`, used by all three call sites instead of the duplicated inline expression, with unit tests covering the `--no-mllm` override, `--mllm` force-on precedence, and the conflicting-flags case (`tests/test_pflash.py::TestResolveEffectiveIsMllm`).

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

## Known limitations (observed directly, 2026-07-10)

Three separate issues surfaced in a single evening of real agentic-coding use (OpenCode, long multi-turn sessions with many tool calls). Documented here with the actual numbers/logs rather than "it feels slow" — two are genuine rapid-mlx gaps; the third is real but not specific to this engine, and is included so nobody budgets hardware/context size against the wrong expectation.

### 1. No admission control for concurrent requests to the same conversation

rapid-mlx batches whatever arrives — it has no mechanism to notice that two in-flight requests are the same growing conversation (the client sent turn N+1 before turn N's response finished) and serialize them. Caught directly via `/metrics`:

```
rapid_mlx_requests_running 3
```

— three overlapping generations for the *same* OpenCode conversation (message counts climbing 165→167→169→171→173→175→176→176→177 across consecutive `[REQUEST]` log lines, each arriving before the previous one's response had been sent). All three compete for the same GPU, degrading throughput for every one of them instead of one finishing before the next starts. On a single-model, single-GPU, single-user backend, concurrent batching has no upside — it only adds contention. (Worked around, for now, one layer up: the [unified-gateway](https://github.com/c-vision/UnifiedLLMGateway) fork serializes all backend-forwarded requests through a mutex before they ever reach rapid-mlx.)

### 2. Metal command-buffer errors abort the whole process, not just the request

Three crashes in three days, identical signature every time — `mlx::core::gpu::check_error` throwing from Metal's async completion-handler dispatch queue, which can't propagate back through Python, so the process dies via `std::terminate()`/`abort()` instead of failing just the one request. rapid-mlx's own `engine_core.py` already documents this as tracked-but-unfixable from this side (issue #353, referencing `mlx-lm#1015`/`ml-explore/mlx`) — the actual fix has to land in MLX itself. mlx **0.32.0** (PR #3523, "catch error in CommandBuffer and poison the events") looks like the real fix; not yet verified against a reproduction of the original crash at time of writing.

### 3. Prefill throughput drops sharply as context grows — largely architectural, not rapid-mlx-specific

Isolated, reproducible measurement (single request, nothing else running, `rapid_mlx_requests_running: 0` confirmed beforehand):

| Prompt size | Time | Throughput |
|---|---|---|
| 21,623 tokens | 30.3s | ~713 tok/s |
| 97,223 tokens | >300s, did not finish | <324 tok/s |

Standard (non-linear-attention) transformer prefill is inherently O(N²) in context length — quadratic compute growth for linear token growth is expected on *any* engine using vanilla attention, not something specific to MLX or this fork. The 21k→97k ratio (~4.5x tokens) predicts almost exactly this kind of falloff under pure O(N²) scaling, so this isn't necessarily a rapid-mlx defect. What genuinely *does* vary between engines is the constant factor — how well-optimized the attention kernels are for a given hardware target (Flash-Attention-style memory access patterns, e.g.) — and that's the open question: whether a more mature, longer-optimized engine (llama.cpp, battle-tested for exactly this workload for years) gets a meaningfully better constant factor on the same hardware. Cross-engine comparison in progress; this section will be updated with the result.

**Practical takeaway regardless of root cause**: don't let a single agentic conversation grow past roughly 20-30k tokens if you need interactive response times — compress/summarize/restart instead of accumulating tool-call history indefinitely. No configuration on rapid-mlx's side (`--prefill-step-size`, `--cache-memory-mb`, TurboQuant) changes this fundamentally; it only shifts where the wall is.

## Original project

This is a fork, not a competing project — most of the engine, the full feature set, and all the heavy lifting come from the original:

**👉 [raullenchai/Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) — read the original README for the complete feature list, benchmarks, and configuration reference.**

## License

[Apache License 2.0](LICENSE)
