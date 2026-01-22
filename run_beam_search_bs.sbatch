#!/bin/bash -l
#SBATCH --account=def-zhu2048
#SBATCH --job-name=bs_qwen_bs
#SBATCH --array=0-29
#SBATCH --gpus-per-node=h100:1
#SBATCH --mem=10G
#SBATCH --time=24:00:00
#SBATCH --output=/home/%u/links/scratch/entropy-gated-branching/slurm/%x_%A_%a.out
#SBATCH --error=/home/%u/links/scratch/entropy-gated-branching/slurm/%x_%A_%a.err

# Route all outputs to scratch (override by exporting EGB_OUTPUT_ROOT before sbatch).
SCRATCH_BASE=${SCRATCH:-$HOME/links/scratch}
export EGB_OUTPUT_ROOT=${EGB_OUTPUT_ROOT:-${SCRATCH_BASE%/}/entropy-gated-branching}
mkdir -p "$EGB_OUTPUT_ROOT" "$EGB_OUTPUT_ROOT/slurm"

# --- config ---
MODELS=("Qwen/Qwen3-1.7B" "Qwen/Qwen3-4B" "Qwen/Qwen3-8B" "llama-1b" "llama-3b" "llama-8b")
DATASETS=("l1" "l2" "gsm" "math" "aime")

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
OUTDIR="beam_search_results/${model_slug}/${EXAM}"

echo "Task ${task}: MODE='BS' MODEL='${MODEL}' EXAM='${EXAM}'"
echo "Output => ${OUTDIR}"

cmd=(
  python3 run_beam_search.py
  --model "$MODEL"
  --exam "$EXAM"
  -o "$OUTDIR"
  --quiet
)

srun "${cmd[@]}"
