#!/bin/bash -l
#SBATCH --account=def-zhu2048
#SBATCH --job-name=bs_qwen_egb
#SBATCH --array=0-23
#SBATCH --gpus-per-node=h100:1
#SBATCH --time=11:50:00
#SBATCH --output=/home/%u/links/scratch/entropy-gated-branching/slurm/%x_%A_%a.out
#SBATCH --error=/home/%u/links/scratch/entropy-gated-branching/slurm/%x_%A_%a.err

source async/bin/activate

# Route all outputs to scratch (override by exporting EGB_OUTPUT_ROOT before sbatch).
SCRATCH_BASE=${SCRATCH:-$HOME/links/scratch}
export EGB_OUTPUT_ROOT=${EGB_OUTPUT_ROOT:-${SCRATCH_BASE%/}/entropy-gated-branching}
mkdir -p "$EGB_OUTPUT_ROOT" "$EGB_OUTPUT_ROOT/slurm"

# --- config ---
# MODELS=("Qwen/Qwen3-1.7B" "Qwen/Qwen3-4B" "Qwen/Qwen3-8B" "llama-1b" "llama-3b" "llama-8b")
MODELS=("llama-1b" "llama-3b" "llama-8b")
# Split larger exams to shorten walltime per array task so the queue moves faster.
DATASETS=(
  "l1" "l1"  # 1800 questions → two halves of 900 each
  "l2"
  "gsm" "gsm"  # 1319 questions → ~660 per slice
  "math" "math"  # 500 questions → two halves of 250
  "aime"
)
STARTS=(
  1 901
  1
  1 661
  1 251
  1
)
LIMITS=(
  900 900
  0
  660 659
  250 250
  0
)

num_models=${#MODELS[@]}
num_sets=${#DATASETS[@]}
task=${SLURM_ARRAY_TASK_ID}

model_idx=$(( task / num_sets ))
data_idx=$(( task % num_sets ))

if (( model_idx >= num_models )) || (( data_idx >= num_sets )); then
  echo "Array index out of range: task=${task} -> model_idx=${model_idx}, data_idx=${data_idx}" >&2
  exit 1
fi

MODEL="${MODELS[$model_idx]}"
EXAM="${DATASETS[$data_idx]}"

model_slug=$(printf "%s" "$MODEL" | tr '[:upper:]' '[:lower:]' | tr '/' '-')
OUTDIR="beam_search_results_egb/${model_slug}/${EXAM}"

echo "Task ${task}: MODE='EGB' MODEL='${MODEL}' EXAM='${EXAM}'"
echo "Output => ${OUTDIR}"

cmd=(
  python3 run_beam_search.py
  --model "$MODEL"
  --exam "$EXAM"
  -o "$OUTDIR"
  --quiet
  --confidence-beam-search
  --entropy-threshold 0.8
  --start "${STARTS[$data_idx]}"
)

limit="${LIMITS[$data_idx]}"
if (( limit > 0 )); then
  cmd+=( --limit "$limit" )
fi

srun "${cmd[@]}"
