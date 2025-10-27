#!/bin/bash
#SBATCH --job-name=cleanup
#SBATCH --error=logs/%j.err
#SBATCH --output=logs/%j.log
#SBATCH --time=03:00:00
#SBATCH --ntasks=1
#SBATCH --mem=1G

# Go to your project directory
cd /scratch/prj/inf_nlg_ai_detection/wcd

# uv cache clean
echo "Deleting .venv ..."
rm -rf .venv

echo "Cleanup done."