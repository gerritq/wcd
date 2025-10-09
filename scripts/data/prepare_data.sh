#!/bin/bash
#SBATCH --job-name=prep
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=00:30:00
#SBATCH --partition=cpu,nmes_cpu
#SBATCH --mem=20GB

uv run prepare_data.py