#!/usr/bin/env python3
"""Pre-download exam datasets so runs can execute offline.

Run this on a node with network access to seed the Hugging Face datasets cache
under ``dataset_cache``.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from datasets import load_dataset


@dataclass(frozen=True)
class DatasetSpec:
    repo_id: str
    config: str
    split: str
    description: str


DATASET_SPECS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        repo_id="openai/gsm8k",
        config="main",
        split="test",
        description="GSM8K test split",
    ),
    DatasetSpec(
        repo_id="HuggingFaceH4/math-500",
        config="default",
        split="test",
        description="Math-500 test split",
    ),
    DatasetSpec(
        repo_id="AI-MO/aimo-validation-aime",
        config="default",
        split="train",
        description="AIMO AIME validation split",
    ),
)

DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[1] / "async-decoding" /  "dataset_cache"


def resolve_cache_dir(cli_value: str | None) -> Path:
    if cli_value:
        return Path(cli_value).expanduser().resolve()
    return DEFAULT_CACHE_DIR


def pin_dataset_cache(
    cache_dir: Path,
    dataset_specs: Iterable[DatasetSpec],
    force_download: bool,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)

    for spec in dataset_specs:
        print(f"\n=== Fetching {spec.description} ({spec.repo_id}:{spec.config}) ===")
        dataset = load_dataset(
            spec.repo_id,
            spec.config,
            split=spec.split,
            cache_dir=str(cache_dir),
        )
        for cache_file in dataset.cache_files:
            print(f"Cached file: {cache_file['filename']}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-download Hugging Face datasets for offline evaluation.")
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Destination directory for the datasets cache (default: dataset_cache).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download of each dataset snapshot.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = resolve_cache_dir(args.cache_dir)

    print(f"Using dataset cache directory: {cache_dir}")
    pin_dataset_cache(
        cache_dir=cache_dir,
        dataset_specs=DATASET_SPECS,
        force_download=args.force,
    )
    print("\nAll requested datasets are present in the cache.")


if __name__ == "__main__":
    main()
