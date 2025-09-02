#!/bin/bash
#SBATCH --job-name=parse-pl
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=02:00:00
#SBATCH --partition=cpu,nmes_cpu
#SBATCH --mem=20GB

uv run parse.py pl