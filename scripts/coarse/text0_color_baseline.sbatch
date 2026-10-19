#!/bin/bash
#SBATCH --job-name="Coarse train"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --mem=30G
#SBATCH --time=0-20:00:00
#SBATCH --mail-type=ALL
#SBATCH --partition=NORMAL
#SBATCH --comment=CVPR
#SBATCH --exclude=node[3-4],node1[0-4],node21
#SBATCH --output=/storage/slurm/shil/output/text2pose/coarse/slurm-%j.out
#SBATCH --error=/storage/slurm/shil/output/text2pose/coarse/slurm-%j.out

# source /usr/stud/kolmet/venv/pyg/bin/activate
conda activate text2loc_sm
module load cuda/11.3.0

# srun python3 -m training.coarse "$@"
python -m training.coarse --batch_size 64 --coarse_embed_dim 256 --shuffle \
  --base_path ./data/k360_30-10_scG_pd10_pc4_spY_all/   \
  --use_features "class"  "color"  "position"  "num" \
  --no_pc_augment \
  --fixed_embedding \
  --epochs 20 \
  --learning_rate 0.0005 \
  --lr_scheduler step \
  --lr_step 4 \
  --lr_gamma 0.85 \
  --temperature 0.1 \
  --hint_type -1 \
  --ranking_loss contrastive \
  --random_discard \
  --hungging_model t5-small \
  --intra_module_num_layers 1 \
  --inter_module_num_layers 1 \
  --inst_num 2 \
  --loss_inst_coeff 1.0 \
  --org_cell \
  --org_text \
  --org_text_coeff 10.0 \
  --pointnet_path checkpoints/pointnet_acc0.86_lr1_p256.pth \
  --folder_name {FOLDER_NAME}

# num work 2
# t5-small
# t5-base
# t5-large
# t5-3b
# t5-11b