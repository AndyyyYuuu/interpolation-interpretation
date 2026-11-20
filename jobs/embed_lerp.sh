#!/bin/bash
#SBATCH --time=0:30:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=linterp-embed
#SBATCH -o jobs/logs/JOB%j.out
#SBATCH -e jobs/logs/JOB%j-err.out

set -uo

cd "${SLURM_SUBMIT_DIR:-$PWD}"
mkdir -p jobs/logs

JOB_ID="${SLURM_JOB_ID:-local}"
echo "job $JOB_ID running on ${SLURM_NODELIST:-localhost}"
echo "start: $(date)"

source .venv/bin/activate

PYTHONPATH="$PWD" .venv/bin/python scripts/embed_lerp.py --job-id "$JOB_ID" "$@"

echo "end: $(date)"
