#!/bin/bash
#SBATCH --job-name="Coarse train"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=36G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --partition=NORMAL
#SBATCH --comment=CVPR
#SBATCH --output=/storage/slurm/shil/output/text2pose/fine/slurm-%j.out
#SBATCH --error=/storage/slurm/shil/output/text2pose/fine/slurm-%j.out

# source /usr/stud/kolmet/venv/pyg/bin/activate
conda activate text2loc_sm
module load cuda/11.3.0

# srun python3 -m training.fine "$@"
python -m training.fine --batch_size 64 --fine_embed_dim 128 --fine_shuffle --shuffle \
  --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/   \
  --use_features "class"  "color"  "position"  "num" \
  --no_pc_augment \
  --fixed_embedding \
  --epochs 35 \
  --learning_rate 0.0003 \
  --fixed_embedding \
  --hungging_model t5-small \
  --regressor_cell all \
  --pmc_prob 0.5 \
  --pointnet_path checkpoints/pointnet_acc0.86_lr1_p256.pth \
  --folder_name {FOLDER_NAME}

# num work 2
# t5-small
# t5-base
# t5-large
# t5-3b
# t5-11b