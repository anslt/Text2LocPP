#!/bin/bash
#SBATCH --job-name="Coarse train"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1,VRAM:24G
#SBATCH --mem=32G
#SBATCH --time=1-0:00:00
#SBATCH --mail-type=ALL
#SBATCH --partition=NORMAL
#SBATCH --comment=CVPR
#SBATCH --output=/storage/slurm/shil/output/text2pose/coarse/slurm-%j.out
#SBATCH --error=/storage/slurm/shil/output/text2pose/coarse/slurm-%j.out

# source /usr/stud/kolmet/venv/pyg/bin/activate
conda activate text2loc_c
module load cuda/11.7.0

# srun python3 -m training.coarse "$@"
python -m training.coarse_aux --batch_size 64 --coarse_embed_dim 256 --shuffle \
  --base_path ./data/pose_dataset_color/unified_3_30-10_gridCells_pd10_pc2_shiftPoses_all_nm-6   \
  --configs color2 \
  --use_features "class"  "color"  "position"  "num" \
  --no_pc_augment \
  --fixed_embedding \
  --epochs 20 \
  --learning_rate 0.0005 \
  --lr_scheduler step \
  --lr_step 4 \
  --lr_gamma 0.7 \
  --temperature 0.1 \
  --hint_type 0 \
  --text_mode 2 \
  --ranking_loss contrastive \
  --aux_hungging_model t5-large \
  --aux_intra_module_num_layers 1 \
  --aux_inter_module_num_layers 1 \
  --aux_intra_proj max \
  --aux_lora_checkpoint {AUX_LORA_CHECKPOINT} \
  --sub_sample \
  --pointnet_path ./checkpoints/pointnet_acc0.86_lr1_p256.pth \
  --add_aux_text \
  --finetune \
  --finetune_path {FINETUNE_PATH} \
  --hungging_model t5-small \
  --intra_module_num_layers 1 \
  --inter_module_num_layers 1 \
  --org_text \
  --folder_name {FOLDER_NAME}

# num work 2
# t5-small
# t5-base
# t5-large
# t5-3b
# t5-11b