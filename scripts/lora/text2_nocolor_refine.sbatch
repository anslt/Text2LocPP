#!/bin/bash
#SBATCH --job-name="Coarse train"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --mem=30G
#SBATCH --time=1-20:00:00
#SBATCH --mail-type=ALL
#SBATCH --partition=NORMAL
#SBATCH --comment=CVPR
#SBATCH --exclude=node1[0-3]
#SBATCH --output=/storage/slurm/shil/output/text2pose/dataset/slurm-%j.out
#SBATCH --error=/storage/slurm/shil/output/text2pose/dataset/slurm-%j.out

#1341443

conda activate text2loc_lora
module load cuda/12.4.1

python training/training_lora2_no_color_refine.py