#!/bin/bash
#SBATCH --job-name=exp1_test
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=00:02:00
#SBATCH --mem=20GB
#SBATCH --reservation=rental_8734
#SBATCH --gres=gpu:1

set -euo pipefail
nvidia-smi
sleep 60