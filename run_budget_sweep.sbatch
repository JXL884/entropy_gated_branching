#!/bin/bash -l
#SBATCH --account=def-zhu2048
#SBATCH --job-name=budget_math
#SBATCH --array=0-19
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=11:00:00
#SBATCH --output=/home/%u/links/scratch/entropy-gated-branching/slurm/%x_%A_%a.out
#SBATCH --error=/home/%u/links/scratch/entropy-gated-branching/slurm/%x_%A_%a.err

set -euo pipefail

source async/bin/activate

# Route all outputs to scratch (override by exporting EGB_OUTPUT_ROOT before sbatch).
SCRATCH_BASE=${SCRATCH:-$HOME/links/scratch}
export EGB_OUTPUT_ROOT=${EGB_OUTPUT_ROOT:-${SCRATCH_BASE%/}/entropy-gated-branching}
mkdir -p "$EGB_OUTPUT_ROOT" "$EGB_OUTPUT_ROOT/slurm"

# Model / algorithm configuration (override via env before submission if desired)
if [[ -n "${MODELS_OVERRIDE:-}" ]]; then
  read -r -a MODELS <<<"${MODELS_OVERRIDE}"
else
  MODELS=("Qwen/Qwen3-1.7B")
fi

if [[ -n "${ALGORITHMS_OVERRIDE:-}" ]]; then
  read -r -a ALGORITHMS <<<"${ALGORITHMS_OVERRIDE}"
else
  ALGORITHMS=(beam egb segs self_consistency)
fi

if [[ -n "${BUDGETS_OVERRIDE:-}" ]]; then
  read -r -a BUDGETS <<<"${BUDGETS_OVERRIDE}"
else
  BUDGETS=(2 4 8 16 32)
fi

EXAM=${EXAM:-math}
DATA_START=${DATA_START:-1}
DATA_LIMIT=${DATA_LIMIT:-500}
QUIET=${QUIET:-1}

# Shared knobs so we keep the notion of "budget = beam width x beam expansions" consistent.
BEAM_EXPANSIONS=${BEAM_EXPANSIONS:-2}
BEAM_MAX_STEP_TOKENS=${BEAM_MAX_STEP_TOKENS:-1024}
EGB_ENTROPY_THRESHOLD=${EGB_ENTROPY_THRESHOLD:-1.0}

SEGS_ITERATIONS=${SEGS_ITERATIONS:-4}
SEGS_TEMPERATURE=${SEGS_TEMPERATURE:-0.7}
SEGS_TOP_P=${SEGS_TOP_P:-0.9}
SEGS_MAX_NEW_TOKENS=${SEGS_MAX_NEW_TOKENS:-1024}

SC_TEMPERATURE=${SC_TEMPERATURE:-0.7}
SC_TOP_P=${SC_TOP_P:-0.9}
SC_TOP_K=${SC_TOP_K:-0}
SC_MAX_NEW_TOKENS=${SC_MAX_NEW_TOKENS:-4096}
SC_USE_EXTRACTOR=${SC_USE_EXTRACTOR:-1}
SC_EXTRACTOR_MAX_NEW_TOKENS=${SC_EXTRACTOR_MAX_NEW_TOKENS:-64}
SC_EXTRACTOR_TEMPERATURE=${SC_EXTRACTOR_TEMPERATURE:-0.0}

num_models=${#MODELS[@]}
num_algorithms=${#ALGORITHMS[@]}
num_budgets=${#BUDGETS[@]}

total_jobs=$(( num_models * num_algorithms * num_budgets ))
if (( SLURM_ARRAY_TASK_ID >= total_jobs )); then
  echo "Array index out of range: task=${SLURM_ARRAY_TASK_ID}, total_jobs=${total_jobs}" >&2
  exit 1
fi

task=${SLURM_ARRAY_TASK_ID}

model_idx=$(( task % num_models ))
budget_idx=$(( (task / num_models) % num_budgets ))
algo_idx=$(( task / (num_models * num_budgets) ))

MODEL="${MODELS[$model_idx]}"
BUDGET="${BUDGETS[$budget_idx]}"
ALGO="${ALGORITHMS[$algo_idx]}"

beam_width=$(( BUDGET / BEAM_EXPANSIONS ))
if (( beam_width < 1 )); then
  beam_width=1
fi
actual_budget=$(( beam_width * BEAM_EXPANSIONS ))
if (( actual_budget != BUDGET )); then
  echo "[warn] Requested budget ${BUDGET}, rounded to ${actual_budget} using expansions=${BEAM_EXPANSIONS}" >&2
fi

model_slug=$(printf "%s" "$MODEL" | tr '[:upper:]' '[:lower:]' | tr '/' '-')
method_slug="${ALGO}"
OUTPUT_DIR="budget_sweep/${method_slug}/budget_${BUDGET}/${model_slug}/${EXAM}"

printf 'Task %d: algo=%s model=%s budget=%s (beam_width=%d, expansions=%d)\n' \
  "$task" "$ALGO" "$MODEL" "$BUDGET" "$beam_width" "$BEAM_EXPANSIONS"
echo "Output => ${OUTPUT_DIR}"

cmd=(python3)

case "$ALGO" in
  beam)
    cmd+=(
      run_beam_search.py
      --model "$MODEL"
      --exam "$EXAM"
      -o "$OUTPUT_DIR"
      --num-beams "$beam_width"
      --num-expansions "$BEAM_EXPANSIONS"
      --max-step-tokens "$BEAM_MAX_STEP_TOKENS"
      --start "$DATA_START"
    )
    if [[ -n "$DATA_LIMIT" && "$DATA_LIMIT" != "0" ]]; then
      cmd+=(--limit "$DATA_LIMIT")
    fi
    ;;
  egb)
    cmd+=(
      run_beam_search.py
      --model "$MODEL"
      --exam "$EXAM"
      -o "$OUTPUT_DIR"
      --num-beams "$beam_width"
      --num-expansions "$BEAM_EXPANSIONS"
      --max-step-tokens "$BEAM_MAX_STEP_TOKENS"
      --confidence-beam-search
      --entropy-threshold "$EGB_ENTROPY_THRESHOLD"
      --start "$DATA_START"
    )
    if [[ -n "$DATA_LIMIT" && "$DATA_LIMIT" != "0" ]]; then
      cmd+=(--limit "$DATA_LIMIT")
    fi
    ;;
  segs)
    cmd+=(
      run_baselines.py
      --model "$MODEL"
      --baseline segs
      --exam "$EXAM"
      -o "$OUTPUT_DIR"
      --segs-beam-width "$beam_width"
      --segs-expansions "$BEAM_EXPANSIONS"
      --segs-iterations "$SEGS_ITERATIONS"
      --segs-temperature "$SEGS_TEMPERATURE"
      --segs-top-p "$SEGS_TOP_P"
      --max-new-tokens "$SEGS_MAX_NEW_TOKENS"
      --start "$DATA_START"
    )
    if [[ -n "$DATA_LIMIT" && "$DATA_LIMIT" != "0" ]]; then
      cmd+=(--limit "$DATA_LIMIT")
    fi
    ;;
  self_consistency)
    cmd+=(
      run_baselines.py
      --model "$MODEL"
      --baseline self_consistency
      --exam "$EXAM"
      -o "$OUTPUT_DIR"
      --sc-samples "$BUDGET"
      --sc-temperature "$SC_TEMPERATURE"
      --sc-top-p "$SC_TOP_P"
      --sc-top-k "$SC_TOP_K"
      --max-new-tokens "$SC_MAX_NEW_TOKENS"
      --start "$DATA_START"
    )
    if [[ -n "$DATA_LIMIT" && "$DATA_LIMIT" != "0" ]]; then
      cmd+=(--limit "$DATA_LIMIT")
    fi
    if (( SC_USE_EXTRACTOR )); then
      cmd+=(
        --sc-use-answer-extractor
        --sc-extractor-max-new-tokens "$SC_EXTRACTOR_MAX_NEW_TOKENS"
        --sc-extractor-temperature "$SC_EXTRACTOR_TEMPERATURE"
      )
    fi
    ;;
  *)
    echo "Unknown algorithm: $ALGO" >&2
    exit 1
    ;;
esac

if (( QUIET )); then
  cmd+=(--quiet)
fi

srun "${cmd[@]}"
