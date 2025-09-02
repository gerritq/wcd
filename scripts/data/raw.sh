#!/bin/bash
#SBATCH --job-name=raw-pl
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=cpu,nmes_cpu

uv run raw.py pl