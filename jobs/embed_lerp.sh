#!/bin/bash
#SBATCH --time=5:00:00
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

PYTHONPATH="$PWD" .venv/bin/python scripts/lerp.py \
    --embed \
    --job-id "$JOB_ID" \
    --models-path "models.csv" \
    "$@"

echo "end: $(date)"
