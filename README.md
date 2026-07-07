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

### 3. Model download tooling (work in progress, **not yet tested end-to-end**)

An experimental parallel-download manager (`vllm_mlx/tools/model_downloader.py` + `model_download_status.py`) intended to speed up and make resumable the process of fetching a model into the alias-path cache, with SHA256 integrity verification. **This is unfinished and unverified** — no automated test coverage yet, it isn't wired into the CLI, and it hasn't been exercised against a real download end-to-end. There's also an earlier, broken draft of the same idea kept around for reference (`model_download_draft.py`) — not used by anything. Treat all of this as a preview, not something to rely on; contributions or bug reports on it are welcome.

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

For the full installation matrix (uv, one-line installer, Homebrew, pip), optional extras (`[vision]`, `[dev]`), and everything else about running and configuring the engine, **see the upstream README** — this fork changes none of that behavior.

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
