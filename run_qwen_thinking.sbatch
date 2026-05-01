#!/bin/bash -l
#SBATCH --account=def-zhu2048
#SBATCH --job-name=qwen3_thinking
#SBATCH --array=0-17
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=11:00:00
#SBATCH --output=/home/%u/links/scratch/entropy-gated-branching/slurm/%x_%A_%a.out
#SBATCH --error=/home/%u/links/scratch/entropy-gated-branching/slurm/%x_%A_%a.err

set -euo pipefail

# Activate the repo env for Python deps.
source async/bin/activate
cd /home/$USER/links/scratch/entropy-gated-branching

# Route outputs to scratch and force Qwen to keep <think> blocks.
SCRATCH_BASE=${SCRATCH:-$HOME/links/scratch}
export EGB_OUTPUT_ROOT=${EGB_OUTPUT_ROOT:-${SCRATCH_BASE%/}/entropy-gated-branching}
export EGB_ENABLE_THINKING=${EGB_ENABLE_THINKING:-1}
mkdir -p "$EGB_OUTPUT_ROOT" "$EGB_OUTPUT_ROOT/slurm"

# Model / dataset / mode grids.
if [[ -n "${MODELS_OVERRIDE:-}" ]]; then
  read -r -a MODELS <<<"${MODELS_OVERRIDE}"
else
  MODELS=("Qwen/Qwen3-1.7B" "Qwen/Qwen3-4B" "Qwen/Qwen3-8B")
fi
# Exams: CFA Level 2 and 40% of math500 (default 200 problems).
EXAMS=("l2" "math")
SEARCH_MODES=("beam" "confidence" "normal")

# Beam/entropy knobs.
NUM_BEAMS=${NUM_BEAMS:-4}
NUM_EXPANSIONS=${NUM_EXPANSIONS:-4}
NUM_STEPS=${NUM_STEPS:-100}
MAX_STEP_TOKENS=${MAX_STEP_TOKENS:-1024}
ENTROPY_THRESHOLD=${ENTROPY_THRESHOLD:-1.5}

# Dataset slicing.
MATH_LIMIT=${MATH_LIMIT:-200}  # 40% of math500 = 200/500
L2_LIMIT=${L2_LIMIT:-}
DATA_START=${DATA_START:-1}
QUIET=${QUIET:-1}

# "Normal" decoding defaults (single-sample nucleus sampling).
NORMAL_TOP_P=${NORMAL_TOP_P:-0.95}
NORMAL_TEMPERATURE=${NORMAL_TEMPERATURE:-0.6}
NORMAL_MAX_NEW_TOKENS=${NORMAL_MAX_NEW_TOKENS:-10240}

num_models=${#MODELS[@]}
num_exams=${#EXAMS[@]}
num_modes=${#SEARCH_MODES[@]}
total_jobs=$(( num_models * num_exams * num_modes ))
task=${SLURM_ARRAY_TASK_ID}

if (( task < 0 || task >= total_jobs )); then
  echo "Array index out of range: task=${task}, total_jobs=${total_jobs}" >&2
  exit 1
fi

group_size=$(( num_exams * num_modes ))
model_idx=$(( task / group_size ))
remainder=$(( task % group_size ))
exam_idx=$(( remainder / num_modes ))
mode_idx=$(( remainder % num_modes ))

MODEL="${MODELS[$model_idx]}"
EXAM="${EXAMS[$exam_idx]}"
MODE="${SEARCH_MODES[$mode_idx]}"

limit=""
case "$EXAM" in
  math) limit="$MATH_LIMIT" ;;
  l2) limit="$L2_LIMIT" ;;
esac

model_slug=$(printf "%s" "$MODEL" | tr '[:upper:]' '[:lower:]' | tr '/' '-')
mode_slug=$(printf "%s" "$MODE" | tr '[:upper:]' '[:lower:]')
OUTDIR="$EGB_OUTPUT_ROOT/qwen_thinking/${mode_slug}/${model_slug}/${EXAM}"

printf 'Task %d: mode=%s model=%s exam=%s start=%s limit=%s\n' \
  "$task" "$MODE" "$MODEL" "$EXAM" "$DATA_START" "${limit:-full}"
echo "Output => ${OUTDIR}"

cmd=(python3)

case "$MODE" in
  beam)
    cmd+=(
      run_beam_search.py
      --model "$MODEL"
      --exam "$EXAM"
      --num-beams "$NUM_BEAMS"
      --num-expansions "$NUM_EXPANSIONS"
      --num-steps "$NUM_STEPS"
      --max-step-tokens "$MAX_STEP_TOKENS"
      --start "$DATA_START"
      -o "$OUTDIR"
    )
    ;;
  confidence)
    cmd+=(
      run_beam_search.py
      --model "$MODEL"
      --exam "$EXAM"
      --num-beams "$NUM_BEAMS"
      --num-expansions "$NUM_EXPANSIONS"
      --num-steps "$NUM_STEPS"
      --max-step-tokens "$MAX_STEP_TOKENS"
      --confidence-beam-search
      --entropy-threshold "$ENTROPY_THRESHOLD"
      --start "$DATA_START"
      -o "$OUTDIR"
    )
    ;;
  normal)
    cmd+=(
      run_baselines.py
      --model "$MODEL"
      --baseline nucleus_best_of_n
      --exam "$EXAM"
      --nucleus-samples 1
      --nucleus-top-p "$NORMAL_TOP_P"
      --nucleus-temperature "$NORMAL_TEMPERATURE"
      --max-new-tokens "$NORMAL_MAX_NEW_TOKENS"
      --start "$DATA_START"
      -o "$OUTDIR"
    )
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 1
    ;;
esac

if [[ -n "$limit" && "$limit" != "0" ]]; then
  cmd+=(--limit "$limit")
fi

if (( QUIET )); then
  cmd+=(--quiet)
fi

echo "Running: ${cmd[*]}"
srun "${cmd[@]}"
