#!/usr/bin/env python3
"""Utility to pre-download all local-only models used by entropy-gated-branching.

Run this on a login node that has network access so the compute nodes can work
offline. The script mirrors the Hugging Face cache layout expected by the
project code under ``async-decoding/model_cache``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from huggingface_hub import snapshot_download


@dataclass(frozen=True)
class ModelSpec:
    repo_id: str



# Keep this list in sync with the hard-coded paths in the project code.
MODEL_SPECS: tuple[ModelSpec, ...] = (
    ModelSpec(repo_id="Qwen/Qwen3-1.7B"),
    ModelSpec(repo_id="Qwen/Qwen3-4B"),
    ModelSpec(repo_id="Qwen/Qwen3-8B"),
    ModelSpec(repo_id="meta-llama/Llama-3.2-1B-Instruct"),
    ModelSpec(repo_id="meta-llama/Llama-3.2-3B-Instruct"),
    ModelSpec(repo_id="meta-llama/Llama-3.1-8B-Instruct"),
    ModelSpec(repo_id="Qwen/Qwen2.5-Math-PRM-7B"),
    ModelSpec(repo_id="RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"),
)
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "entropy-gated-branching" / "model_cache"

def resolve_cache_dir(cli_value: str | None) -> Path:
    """Resolve the model cache directory, defaulting to async-decoding/model_cache."""
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    return DEFAULT_CACHE_DIR


def download_all_models(cache_dir: Path, specs: Iterable[ModelSpec], skip_existing: bool) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)

    for spec in specs:
        print(f"\n=== Fetching {spec.repo_id} ===")
        try:
            snapshot_download(
                repo_id=spec.repo_id,
                cache_dir=str(cache_dir),
                local_files_only=skip_existing,
                token="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"
            )
            if skip_existing:
                print("Already present in cache; nothing to download.")
            else:
                print("Download complete.")
        except Exception as exc:  # noqa: BLE001 - surface unexpected download issues clearly
            if not skip_existing:
                raise
            print(f"Missing local files or cache mismatch ({exc}); retrying with network download enabled...")
            snapshot_download(
                repo_id=spec.repo_id,
                cache_dir=str(cache_dir),
                local_files_only=False,
                token="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"
            )
            print("Download complete.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-download Hugging Face models for offline inference.")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Directory that should hold the Hugging Face cache layout (default: async-decoding/model_cache).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-downloading the snapshots even if they already exist locally.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = resolve_cache_dir(args.cache_dir)

    print(f"Using cache directory: {cache_dir}")
    skip_existing = not args.force

    download_all_models(cache_dir=cache_dir, specs=MODEL_SPECS, skip_existing=skip_existing)

    print("\nAll requested models are present in the cache.")


if __name__ == "__main__":
    main()
