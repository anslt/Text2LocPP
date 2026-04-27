# Text2Loc_v2

This is the codebase for Text2Loc++, the extension of Text2Loc (CVPR 2024).

This repository is a **demo repo** that focuses on running Text2Loc++ inference on the Toronto dataset under three language complexity settings:
- Simple
- Moderate
- Complex

## Environment Setup

This repo uses **two different inference environments**:

- `simple` / `moderate` inference: use the base environment with `requirements.txt`
- `complex` inference: use a separate environment with `requirements2.txt`

### Environment A (Simple / Moderate)

```bash
conda create -n text2loc_sm python=3.10 -y
conda activate text2loc_sm

# Install matching PyTorch / torchvision / torchaudio
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install other dependencies
CC=/usr/bin/gcc-9 CXX=/usr/bin/g++-9 pip install -r requirements.txt
```

### Environment B (Complex)

```bash
conda create -n text2loc_complex python=3.10 -y
conda activate text2loc_complex
CC=/usr/bin/gcc-9 CXX=/usr/bin/g++-9 pip install -r requirements2.txt \
  -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

## Dataset (Toronto)

Download the Toronto demo dataset from:

- Toronto dataset: [Here](https://huggingface.co/datasets/sltttt/Text2Loc)

Place the data under:

The final directory structure should be:
```
│Text2Loc/
├──dataloading/
├──datapreparation/
├──data/
│   ├──pose_dataset_color
│       ├──unified_3_30-10_gridCells_pd10_pc2_shiftPoses_all_nm-6
│           ├──cells/
│               ├──toronto.pkl
│               ├──...
│           ├──poses/
│               ├──toronto.pkl
│               ├──...
│           ├──direction/
│               ├──toronto.json
│               ├──...
│           ├──texts1/
│               ├──toronto.json
│               ├──...
│           ├──gpt_texts1/
│               ├──toronto.json
│               ├──...
│           ├──texts4/
│               ├──toronto.json
│               ├──...
│           ├──gpt_texts4/
│               ├──toronto.json
│               ├──...
├──checkpoints/
│   ├──pointnet_acc0.86_lr1_p256.pth
├──...
```

## Checkpoints

Download pretrained backbone the pretrained object backbone [HERE](https://drive.google.com/file/d/1j2q67tfpVfIbJtC1gOWm7j8zNGhw5J9R/view?usp=drive_link) and pretrained checkpoints [Here](https://huggingface.co/datasets/sltttt/Text2Loc) and place them at:

```text
./checkpoints/coarse_baseline_text0.pth
./checkpoints/coarse_baseline_text1.pth
./checkpoints/coarse_baseline_text2.pth
./checkpoints/fine_baseline_text0.pth
./checkpoints/fine_baseline_text1.pth
./checkpoints/lora
```

## Inference Demo (Toronto)

The demo inference code is provided for three language complexity settings via `--text_mode`:
- `0`: simple
- `1`: moderate
- `2`: complex

### Simple

```bash
conda activate text2loc_sm
python -m evaluation.pipeline --coarse_embed_dim 256 --fine_embed_dim 128 \
  --base_path ./data/pose_dataset_color/unified_3_30-10_gridCells_pd10_pc2_shiftPoses_all_nm-6 \
  --use_features "class"  "color"  "position"  "num" \
  --configs color \
  --no_pc_augment \
  --no_pc_augment_fine \
  --hint_type -1 \
  --text_mode 0 \
  --hungging_model t5-small \
  --intra_module_num_layers 1 \
  --inter_module_num_layers 1 \
  --fine_hungging_model t5-small \
  --use_test_set \
  --path_coarse ./checkpoints/coarse_baseline_text0.pth \
  --path_fine ./checkpoints/fine_baseline_text0.pth

```

### Moderate

```bash
conda activate text2loc_sm
python -m evaluation.pipeline --coarse_embed_dim 256 --fine_embed_dim 128 \
  --base_path ./data/pose_dataset_color/unified_3_30-10_gridCells_pd10_pc2_shiftPoses_all_nm-6 \
  --use_features "class"  "color"  "position"  "num" \
  --configs color \
  --no_pc_augment \
  --no_pc_augment_fine \
  --hint_type 0 \
  --text_mode 1 \
  --add_aux_text \
  --aux_hungging_model t5-large \
  --aux_intra_module_num_layers 1 \
  --aux_inter_module_num_layers 1 \
  --aux_intra_proj max \
  --hungging_model t5-small \
  --intra_module_num_layers 1 \
  --inter_module_num_layers 1 \
  --fine_hungging_model t5-small \
  --use_test_set \
  --fine_hungging_model t5-small \
  --path_coarse ./checkpoints/coarse_baseline_text1.pth \
  --path_fine ./checkpoints/fine_baseline_text1.pth

```

### Complex

```bash
conda activate text2loc_complex
python -m evaluation.coarse_aux --coarse_embed_dim 256 --fine_embed_dim 128 \
  --base_path ./data/pose_dataset_color/unified_3_30-10_gridCells_pd10_pc2_shiftPoses_all_nm-6  \
  --use_features "class"  "color"  "position"  "num" \
  --configs color \
  --no_pc_augment \
  --hint_type 0 \
  --text_mode 2 \
  --add_aux_text \
  --aux_hungging_model t5-large \
  --aux_intra_module_num_layers 1 \
  --aux_inter_module_num_layers 1 \
  --aux_intra_proj max \
  --aux_lora_checkpoint ./checkpoints/lora/text2/t5-large/checkpoint-30788 \
  --hungging_model t5-small \
  --intra_module_num_layers 1 \
  --inter_module_num_layers 1 \
  --fine_hungging_model t5-large \
  --fine_lora_checkpoint ./checkpoints/lora/text2/t5-large/checkpoint-30788 \
  --use_test_set \
  --path_coarse ./checkpoints/fine_baseline_text2.pth


```
