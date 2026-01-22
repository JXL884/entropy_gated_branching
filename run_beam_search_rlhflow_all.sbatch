#!/bin/bash -l
#SBATCH --account=def-zhu2048
#SBATCH --job-name=egb_rlhflow_all
#SBATCH --array=0-59
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=10:00:00
#SBATCH --output=/home/%u/links/scratch/entropy-gated-branching/slurm/%x_%A_%a.out
#SBATCH --error=/home/%u/links/scratch/entropy-gated-branching/slurm/%x_%A_%a.err

source async/bin/activate

SCRATCH_BASE=${SCRATCH:-$HOME/links/scratch}
export EGB_OUTPUT_ROOT=${EGB_OUTPUT_ROOT:-${SCRATCH_BASE%/}/entropy-gated-branching}
mkdir -p "$EGB_OUTPUT_ROOT" "$EGB_OUTPUT_ROOT/slurm"

DEFAULT_MODELS=("Qwen/Qwen3-1.7B" "llama-1b" "Qwen/Qwen3-4B"  "llama-3b" "Qwen/Qwen3-8B" "llama-8b")
if [[ -n ${MODEL_LIST:-} ]]; then
  IFS=' ' read -r -a MODELS <<< "${MODEL_LIST}"
elif [[ -n ${MODEL:-} ]]; then
  MODELS=("${MODEL}")
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

DEFAULT_EXAMS=("l1" "l2" "gsm" "math" "aime")
if [[ -n ${EXAM_LIST:-} ]]; then
  IFS=' ' read -r -a EXAMS <<< "${EXAM_LIST}"
elif [[ -n ${EXAM:-} ]]; then
  EXAMS=("${EXAM}")
else
  EXAMS=("${DEFAULT_EXAMS[@]}")
fi

DEFAULT_SEARCH_MODES=("beam" "confidence")
if [[ -n ${SEARCH_MODE_LIST:-} ]]; then
  IFS=' ' read -r -a SEARCH_MODES <<< "${SEARCH_MODE_LIST}"
else
  SEARCH_MODES=("${DEFAULT_SEARCH_MODES[@]}")
fi

NUM_BEAMS=${NUM_BEAMS:-4}
NUM_EXPANSIONS=${NUM_EXPANSIONS:-4}
NUM_STEPS=${NUM_STEPS:-100}
MAX_STEP_TOKENS=${MAX_STEP_TOKENS:-1024}
DATA_LIMIT=${DATA_LIMIT:-}
DATA_START=${DATA_START:-1}
QUIET=${QUIET:-1}
ENTROPY_THRESHOLD=${ENTROPY_THRESHOLD:-1.0}
SCORER_MODEL=${SCORER_MODEL:-rlhflow}

NUM_MODELS=${#MODELS[@]}
NUM_EXAMS=${#EXAMS[@]}
NUM_SEARCH_MODES=${#SEARCH_MODES[@]}
TOTAL_JOBS=$(( NUM_MODELS * NUM_EXAMS * NUM_SEARCH_MODES ))
TASK_ID=${SLURM_ARRAY_TASK_ID}

if (( NUM_MODELS == 0 || NUM_EXAMS == 0 || NUM_SEARCH_MODES == 0 )); then
  echo "MODELS, EXAMS, and SEARCH_MODES must be non-empty" >&2
  exit 1
fi

if (( TASK_ID < 0 || TASK_ID >= TOTAL_JOBS )); then
  echo "Array index out of range: task=${TASK_ID} (total jobs=${TOTAL_JOBS})" >&2
  exit 1
fi

group_size=$(( NUM_EXAMS * NUM_SEARCH_MODES ))
model_idx=$(( TASK_ID / group_size ))
remainder=$(( TASK_ID % group_size ))
exam_idx=$(( remainder / NUM_SEARCH_MODES ))
mode_idx=$(( remainder % NUM_SEARCH_MODES ))

MODEL="${MODELS[$model_idx]}"
EXAM="${EXAMS[$exam_idx]}"
SEARCH_MODE="${SEARCH_MODES[$mode_idx]}"

model_slug=$(printf "%s" "$MODEL" | tr '[:upper:]' '[:lower:]' | tr '/' '-')
mode_slug=$(printf "%s" "$SEARCH_MODE" | tr '[:upper:]' '[:lower:]')

echo "Task ${TASK_ID}: MODEL='${MODEL}' EXAM='${EXAM}' MODE='${SEARCH_MODE}'"
echo "MODELS=${MODELS[*]}"; echo "EXAMS=${EXAMS[*]}"; echo "SEARCH_MODES=${SEARCH_MODES[*]}"

OUTDIR="rlhflow_runs/${mode_slug}/${model_slug}/${EXAM}"
cmd=(
  python3 run_beam_search.py
  --model "$MODEL"
  --exam "$EXAM"
  --num-beams "$NUM_BEAMS"
  --num-expansions "$NUM_EXPANSIONS"
  --num-steps "$NUM_STEPS"
  --max-step-tokens "$MAX_STEP_TOKENS"
  --scorer-model "$SCORER_MODEL"
  -o "$OUTDIR"
)

if [[ "$SEARCH_MODE" == "confidence" ]]; then
  cmd+=(--confidence-beam-search --entropy-threshold "$ENTROPY_THRESHOLD")
fi

if [[ -n "$DATA_LIMIT" ]]; then
  cmd+=(--limit "$DATA_LIMIT")
fi

if [[ -n "$DATA_START" ]]; then
  cmd+=(--start "$DATA_START")
fi

if (( QUIET )); then
  cmd+=(--quiet)
fi

echo "Running: ${cmd[*]}"
srun "${cmd[@]}"
