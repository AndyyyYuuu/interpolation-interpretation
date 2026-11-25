#!/bin/bash
#SBATCH --time=0:30:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=linterp-embed
#SBATCH -o jobs/logs/JOB%j.out

set -uo

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p jobs/logs

JOB_ID="${SLURM_JOB_ID:-local}"
echo "job $JOB_ID running on ${SLURM_NODELIST:-localhost}"
echo "start: $(date)"

source .venv/bin/activate

PYTHONPATH="$PWD" .venv/bin/python scripts/embed_lerp.py \
    --job-id "$JOB_ID" \
    --model-name \
        HuggingFaceTB/SmolLM2-135M \
        HuggingFaceTB/SmolLM2-360M \
        HuggingFaceTB/SmolLM2-1.7B \
        Qwen/Qwen2.5-0.5B \
        Qwen/Qwen2.5-1.5B \
        meta-llama/Llama-3.2-1B \
        google/gemma-3-1b-pt \
        Qwen/Qwen2.5-3B \
        meta-llama/Llama-3.2-3B \
        google/gemma-3-4b-pt \
        google/gemma-2-2b \
        microsoft/Phi-3.5-mini \
        Qwen/Qwen2.5-7B \
        meta-llama/Llama-3.1-8B \
        mistralai/Mistral-7B-v0.3 \
        allenai/OLMo-2-1124-7B \
        google/gemma-2-9b \
        google/gemma-3-9b-pt \
    "$@"

echo "end: $(date)"
