#!/bin/bash
#SBATCH --job-name=uk-id
#SBATCH --output=../../logs/%j.out
#SBATCH --error=../../logs/%j.err
#SBATCH --time=10:00:00
#SBATCH --mem=40GB
#SBATCH --partition=cpu,nmes_cpu

uv run raw.py