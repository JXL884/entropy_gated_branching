#!/bin/bash -l
#SBATCH --account=def-zhu2048
#SBATCH --job-name=egb_entropy_ablation
#SBATCH --array=0-83  # Adjust if overriding model/threshold/exam counts.
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=10:00:00
#SBATCH --output=/home/%u/links/scratch/entropy-gated-branching/slurm/%x_%A_%a.out
#SBATCH --error=/home/%u/links/scratch/entropy-gated-branching/slurm/%x_%A_%a.err

source async/bin/activate

# Route all outputs to scratch (override by exporting EGB_OUTPUT_ROOT before sbatch).
SCRATCH_BASE=${SCRATCH:-$HOME/links/scratch}
export EGB_OUTPUT_ROOT=${EGB_OUTPUT_ROOT:-${SCRATCH_BASE%/}/entropy-gated-branching}
mkdir -p "$EGB_OUTPUT_ROOT" "$EGB_OUTPUT_ROOT/slurm"

# Optional overrides (export before submitting, e.g., MODEL_LIST="model1 model2" sbatch ...)
DEFAULT_MODELS=("Qwen/Qwen3-1.7B" "llama-1b" "Qwen/Qwen3-4B"  "llama-3b" "Qwen/Qwen3-8B" "llama-8b")
if [[ -n ${MODEL_LIST:-} ]]; then
  IFS=' ' read -r -a MODELS <<< "${MODEL_LIST}"
elif [[ -n ${MODEL:-} ]]; then
  MODELS=("${MODEL}")
else
  MODELS=("${DEFAULT_MODELS[@]}")
fi

DEFAULT_EXAMS=("math" "l2")  # l2 corresponds to the CFA Level II dataset.
if [[ -n ${EXAM_LIST:-} ]]; then
  IFS=' ' read -r -a EXAMS <<< "${EXAM_LIST}"
elif [[ -n ${EXAM:-} ]]; then
  EXAMS=("${EXAM}")
else
  EXAMS=("${DEFAULT_EXAMS[@]}")
fi
NUM_BEAMS=${NUM_BEAMS:-4}
NUM_EXPANSIONS=${NUM_EXPANSIONS:-4}
NUM_STEPS=${NUM_STEPS:-80}
MAX_STEP_TOKENS=${MAX_STEP_TOKENS:-1024}
DATA_LIMIT=${DATA_LIMIT:-}
DATA_START=${DATA_START:-1}
QUIET=${QUIET:-1}

DEFAULT_THRESHOLDS=("0.5" "1.0" "1.5" "2.0" "2.5" "3.0" "4.0")
if [[ -n ${ENTROPY_THRESHOLDS:-} ]]; then
  IFS=' ' read -r -a THRESHOLDS <<< "${ENTROPY_THRESHOLDS}"
else
  THRESHOLDS=("${DEFAULT_THRESHOLDS[@]}")
fi

NUM_MODELS=${#MODELS[@]}
NUM_THRESHOLDS=${#THRESHOLDS[@]}
NUM_EXAMS=${#EXAMS[@]}
TOTAL_JOBS=$(( NUM_MODELS * NUM_THRESHOLDS * NUM_EXAMS ))
TASK_ID=${SLURM_ARRAY_TASK_ID}

if (( NUM_MODELS == 0 || NUM_THRESHOLDS == 0 || NUM_EXAMS == 0 )); then
  echo "MODELS, THRESHOLDS, and EXAMS must be non-empty" >&2
  exit 1
fi

if (( TASK_ID < 0 || TASK_ID >= TOTAL_JOBS )); then
  echo "Array index out of range: task=${TASK_ID} (total jobs=${TOTAL_JOBS})" >&2
  exit 1
fi

group_size=$(( NUM_THRESHOLDS * NUM_EXAMS ))
model_idx=$(( TASK_ID / group_size ))
remainder=$(( TASK_ID % group_size ))
thresh_idx=$(( remainder / NUM_EXAMS ))
exam_idx=$(( remainder % NUM_EXAMS ))

MODEL="${MODELS[$model_idx]}"
ENTROPY_THRESHOLD="${THRESHOLDS[$thresh_idx]}"
EXAM="${EXAMS[$exam_idx]}"

model_slug=$(printf "%s" "$MODEL" | tr '[:upper:]' '[:lower:]' | tr '/' '-')
thresh_slug=$(printf "%s" "$ENTROPY_THRESHOLD" | tr '.' 'p')

echo "Task ${TASK_ID}: MODEL='${MODEL}' EXAM='${EXAM}' ENTROPY_THRESHOLD='${ENTROPY_THRESHOLD}'"
echo "MODELS=${MODELS[*]}"; echo "THRESHOLDS=${THRESHOLDS[*]}"; echo "EXAMS=${EXAMS[*]}"

OUTDIR="beam_search_entropy_ablation/${model_slug}/${EXAM}/thresh_${thresh_slug}"
echo "Saving under ${OUTDIR}"
cmd=(
  python3 run_beam_search.py
  --model "$MODEL"
  --exam "$EXAM"
  --entropy-threshold "$ENTROPY_THRESHOLD"
  --num-beams "$NUM_BEAMS"
  --num-expansions "$NUM_EXPANSIONS"
  --num-steps "$NUM_STEPS"
  --max-step-tokens "$MAX_STEP_TOKENS"
  -o "$OUTDIR"
  --confidence-beam-search
)

if [[ -n "$DATA_LIMIT" ]]; then
  cmd+=(--limit "$DATA_LIMIT")
fi

if [[ -n "$DATA_START" ]]; then
  cmd+=(--start "$DATA_START")
fi

if (( QUIET )); then
  cmd+=(--quiet)
fi

srun "${cmd[@]}"
