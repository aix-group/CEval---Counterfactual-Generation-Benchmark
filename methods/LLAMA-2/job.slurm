#!/bin/bash
#SBATCH --job-name=imdb_llama
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=1-0
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=4000M
#SBATCH --account=fb_12
#SBATCH --partition=owner_fb12
#SBATCH --output=imdb_llama.out
bash methods/LLAMA-2/script.sh