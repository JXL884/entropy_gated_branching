#!/usr/bin/env python3
"""Helper script to run eval.py across entropy ablation outputs."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from paths import ensure_dir, get_output_root, resolve_in_output_root

# Map directory names to eval.py dataset choices
DATASET_NAME_MAP: Dict[str, str] = {
    "aime": "aime",
    "gsm": "gsm8k",
    "l1": "cfa-l1",
    "l2": "cfa-l2",
    "math": "math500",
}
DATASET_DISPLAY_MAP: Dict[str, str] = {v: k for k, v in DATASET_NAME_MAP.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch runner for eval.py over entropy ablation outputs."
    )
    default_root = get_output_root() / "beam_search_entropy_ablation"
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root,
        help=(
            "Root directory containing entropy ablation outputs (default: "
            "beam_search_entropy_ablation under the output root)."
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["parser", "llm"],
        default="llm",
        help="Evaluation mode passed through to eval.py (default: llm).",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Optional list of model directory names to evaluate.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        help="Optional list of dataset names to evaluate (leaf folders like 'math', 'gsm').",
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        help="Optional list of threshold subdirectory names (e.g. thresh_1p0).",
    )
    parser.add_argument(
        "--llm-model",
        default="google/gemini-2.5-flash",
        help="LLM model name for llm mode (ignored for parser mode).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=100,
        help="Number of concurrent workers for llm mode (ignored for parser mode).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Optional directory to mirror evaluation outputs instead of writing "
            "alongside inputs."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Only print the eval.py commands without executing them.",
    )
    return parser.parse_args()


def iter_result_dirs(
    root: Path,
    models: Iterable[str] | None,
    datasets: Iterable[str] | None,
    thresholds: Iterable[str] | None,
) -> Iterable[Tuple[str, str, Optional[str], Path]]:
    """Yield (model_name, dataset_name, threshold_name, path) for each eval target."""
    if not root.exists():
        raise FileNotFoundError(f"Results root '{root}' does not exist.")

    model_filter = set(models) if models else None
    dataset_filter = set(datasets) if datasets else None
    threshold_filter = set(thresholds) if thresholds else None

    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        model_name = model_dir.name
        if model_filter and model_name not in model_filter:
            continue

        for dataset_dir in sorted(p for p in model_dir.iterdir() if p.is_dir()):
            dataset_name = dataset_dir.name
            dataset_key = DATASET_NAME_MAP.get(dataset_name)
            if dataset_key is None:
                print(f"[skip] {model_name}/{dataset_name}: no dataset mapping defined.")
                continue

            if dataset_filter and dataset_name not in dataset_filter and dataset_key not in dataset_filter:
                continue

            json_files = list(dataset_dir.glob("*.json"))
            if json_files:
                if threshold_filter:
                    # User explicitly requested thresholds; skip flat outputs.
                    continue
                yield model_name, dataset_key, None, dataset_dir
                continue

            threshold_dirs = sorted(p for p in dataset_dir.iterdir() if p.is_dir())
            if not threshold_dirs:
                print(f"[skip] {model_name}/{dataset_name}: no JSON files or threshold subfolders found.")
                continue

            for threshold_dir in threshold_dirs:
                threshold_name = threshold_dir.name
                if threshold_filter and threshold_name not in threshold_filter:
                    continue
                has_json = any(threshold_dir.glob("*.json"))
                if not has_json:
                    print(f"[skip] {model_name}/{dataset_name}/{threshold_name}: no JSON files found.")
                    continue
                yield model_name, dataset_key, threshold_name, threshold_dir


def build_command(
    eval_py: Path,
    dataset_dir: Path,
    dataset_key: str,
    mode: str,
    llm_model: str,
    num_workers: int,
    output_root: Path | None,
    root: Path,
) -> List[str]:
    """Compose the eval.py command for a given dataset directory."""
    cmd: List[str] = [
        sys.executable,
        str(eval_py),
        "--input-dir",
        str(dataset_dir),
        "--dataset",
        dataset_key,
        "--mode",
        mode,
    ]

    if mode == "llm":
        cmd.extend([
            "--llm-model",
            llm_model,
            "--num-workers",
            str(num_workers),
        ])

    if output_root is not None:
        rel_path = dataset_dir.relative_to(root)
        target_dir = output_root / rel_path
        ensure_dir(target_dir)
        output_file = target_dir / "evaluation_summary.json"
        cmd.extend(["--output-file", str(output_file)])

    return cmd


def main() -> int:
    args = parse_args()
    root = resolve_in_output_root(args.root)
    eval_py = Path(__file__).resolve().parent / "eval.py"
    output_root = resolve_in_output_root(args.output_root) if args.output_root else None

    successes: List[Tuple[str, str, Optional[str]]] = []
    failures: List[Tuple[str, str, Optional[str], int]] = []

    for model_name, dataset_key, threshold_name, dataset_dir in iter_result_dirs(
        root,
        args.models,
        args.datasets,
        args.thresholds,
    ):
        cmd = build_command(
            eval_py,
            dataset_dir,
            dataset_key,
            args.mode,
            args.llm_model,
            args.num_workers,
            output_root,
            root,
        )

        dataset_display = DATASET_DISPLAY_MAP[dataset_key]
        if threshold_name:
            pretty_target = f"{model_name}/{dataset_display}/{threshold_name}"
        else:
            pretty_target = f"{model_name}/{dataset_display}"
        print(f"[run] {pretty_target}: {' '.join(cmd)}")

        if args.dry_run:
            continue

        try:
            subprocess.run(cmd, check=True)
            successes.append((model_name, dataset_key, threshold_name))
        except subprocess.CalledProcessError as exc:
            print(f"[error] {pretty_target} failed with exit code {exc.returncode}.")
            failures.append((model_name, dataset_key, threshold_name, exc.returncode))

    print()
    print(f"Completed eval.py runs: {len(successes)} succeeded, {len(failures)} failed.")
    if successes:
        succeeded = ", ".join(
            f"{m}/{DATASET_DISPLAY_MAP[d]}"
            + (f"/{t}" if t else "")
            for m, d, t in successes
        )
        print(f"  Success: {succeeded}")
    if failures:
        failed = ", ".join(
            f"{m}/{DATASET_DISPLAY_MAP[d]}"
            + (f"/{t}" if t else "")
            + f" (exit {code})"
            for m, d, t, code in failures
        )
        print(f"  Failures: {failed}")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
