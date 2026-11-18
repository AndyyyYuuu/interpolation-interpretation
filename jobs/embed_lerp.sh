#!/bin/bash
#SBATCH --time=0:30:00
#SBATCH --mem=16GB
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --job-name=linterp-embed
#SBATCH -o jobs/logs/JOB%j.out
#SBATCH -e jobs/logs/JOB%j-err.out

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p jobs/logs

echo "job $SLURM_JOB_ID running on $SLURM_NODELIST"
echo "start: $(date)"

source .venv/bin/activate

PYTHONPATH="$PWD" python scripts/embed_lerp.py --job-id "$SLURM_JOB_ID" "$@"

echo "end: $(date)"
