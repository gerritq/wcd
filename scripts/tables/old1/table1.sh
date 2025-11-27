#!/bin/bash
#SBATCH --job-name=table1
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=00:20:00
#SBATCH --partition=nmes_cpu,cpu
#SBATCH --mem=10GB

uv run table1.py