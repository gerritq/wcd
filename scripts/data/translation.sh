#!/bin/bash
#SBATCH --job-name=translations
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=05:30:00
#SBATCH --partition=cpu,nmes_cpu
#SBATCH --mem=10GB

uv run translation.py
