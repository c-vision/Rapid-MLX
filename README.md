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

A parallel-download manager (`vllm_mlx/tools/model_downloader.py` + `model_download_status.py`) that fetches a full HuggingFace model repo (config, tokenizer, weight shards — not just one file) into a target directory, with resume support and SHA256 verification of LFS-tracked files. File transfer itself is delegated to `huggingface_hub.snapshot_download` (already a core dependency) rather than reimplementing HTTP range-request resume logic — `snapshot_download` already does that correctly, including per-file parallelism (default 8 workers).

**Full example — download into a folder of your choice, then register it under an alias of your choice:**

```python
from vllm_mlx.tools.model_downloader import download_model

# 1. Download a model. `dest_dir` can be anywhere — `~` is expanded, and the
#    model lands in `dest_dir/<repo-name>` (here: ~/my-models/SmolLM2-135M-Instruct-8bit).
success, path = download_model(
    "mlx-community/SmolLM2-135M-Instruct-8bit",
    dest_dir="~/my-models",
)
if not success:
    raise SystemExit("Download failed — see the printed error above")
print(f"Downloaded to: {path}")
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

Interrupted downloads resume correctly: `huggingface_hub` tracks partial files under `<dest_dir>/<repo-name>/.cache/huggingface/download/*.incomplete`, and re-running the same `download_model(...)` call skips whatever's already complete and only re-fetches what's missing — verified by killing a download mid-transfer and re-running it, which picked up exactly where it left off and passed the SHA256 check afterward.

Verified end-to-end, literally running the three steps above: fresh download to a custom folder, kill-and-resume, register a custom alias with a path override, serve it, and get a real completion back through the API.

It isn't wired into the CLI yet (`rapid-mlx ds4 download ...`-style commands from earlier drafts of this README were aspirational, not implemented) — for now it's a Python API, not a subcommand. There's also an earlier, broken draft of the same idea kept around for reference (`model_download_draft.py`) — not used by anything, not exported, not maintained.

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
