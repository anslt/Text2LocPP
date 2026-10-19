#!/bin/bash
#SBATCH --job-name="Coarse train"
#SBATCH --nodes=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1,VRAM:48G
#SBATCH --mem=32G
#SBATCH --time=1-20:00:00
#SBATCH --mail-type=ALL
#SBATCH --partition=NORMAL
#SBATCH --comment=CVPR
#SBATCH --output=/storage/slurm/shil/output/text2pose/coarse/slurm-%j.out
#SBATCH --error=/storage/slurm/shil/output/text2pose/coarse/slurm-%j.out

# source /usr/stud/kolmet/venv/pyg/bin/activate
conda activate text2loc_sm
module load cuda/11.3.0

# srun python3 -m training.coarse "$@"
python -m training.coarse --batch_size 64 --coarse_embed_dim 256 --shuffle \
  --base_path ./data/pose_dataset_no_color/unified_3_30-10_gridCells_pd10_pc2_shiftPoses_all_nm-6 \
  --use_features "class" "position"  "num" \
  --no_color \
  --no_color2 --no_color0 \
  --no_pc_augment \
  --fixed_embedding \
  --epochs 20 \
  --learning_rate 0.0005 \
  --lr_scheduler step \
  --lr_step 4 \
  --lr_gamma 0.7 \
  --temperature 0.1 \
  --hint_type -1 \
  --ranking_loss contrastive \
  --hungging_model t5-small \
  --configs no_color2 \
  --random_discard \
  --intra_module_num_layers 1 \
  --inter_module_num_layers 1 \
  --inst_num 2 \
  --loss_inst_coeff 1.0 \
  --org_cell \
  --org_text \
  --org_text_coeff 10.0 \
  --folder_name {FOLDER_NAME}

# python -m training.coarse --batch_size 64 --coarse_embed_dim 256 --shuffle \
#   --base_path ./data/pose_dataset_color/unified_3_30-10_gridCells_pd10_pc2_shiftPoses_all_nm-6 \
#   --use_features "class"  "color"  "position"  "num" \
#   --no_pc_augment \
#   --fixed_embedding \
#   --epochs 20 \
#   --learning_rate 0.0005 \
#   --lr_scheduler step \
#   --lr_step 4 \
#   --lr_gamma 0.7 \
#   --temperature 0.1 \
#   --hint_type -1 \
#   --ranking_loss contrastive \
#   --hungging_model t5-small \
#   --configs color \
#   --random_discard \
#   --intra_module_num_layers 1 \
#   --inter_module_num_layers 1 \
#   --intra_mlp \
#   --inst_num 2 \
#   --loss_inst_coeff 1.0 \
#   --fix_lang \
#   --finetune \
#   --finetune_path ./checkpoints/k360_30-10_scG_pd10_pc4_spY_all/text0-t5-small_1_1_4_0.72025_01_15_23_53_57/coarse_contN_epoch17_acc0.617_ecl0_eco0_p256_npa1_loss-contrastive_f-class-color-position-num.pth \
#   --folder_name finetune_color_text0-t5-small_1_1_4_0.7_fix_lang_`date +"%Y_%m_%d_%H_%M_%S"` >> 2.out

# num work 2
# t5-small
# t5-base
# t5-large
# t5-3b
# t5-11b