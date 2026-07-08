# Text2Loc++

This repository is the official implementation of our paper:

[Text2Loc++: Generalizing 3D Point Cloud Localization from Natural Language](https://arxiv.org/abs/2511.15308)

It extends [Text2Loc: 3D Point Cloud Localization from Natural Language (CVPR 2024)](https://arxiv.org/abs/2311.15977) and supports three language complexity settings:
- Simple
- Moderate
- Complex

## Environment Setup

This repo uses **three environments**:

- `simple` / `moderate` (text0 / text1): environment **`text2loc_sm`** with `requirements.txt`
- `complex` (text2): environment **`text2loc_c`** with `requirements2.txt` (torch 2.0.0+cu117; loading the text2 LoRA requires CUDA ≥ 11.7)
- LoRA training (text2 t5-large LoRA): environment **`text2loc_lora`** with `requirements3.txt` (Python 3.11, torch 2.6.0+cu124)

### Environment A (Simple / Moderate, text2loc_sm)

```bash
conda create -n text2loc_sm python=3.10 -y
conda activate text2loc_sm

# Install matching PyTorch / torchvision / torchaudio
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 cudatoolkit=11.3 -c pytorch
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 --extra-index-url https://download.pytorch.org/whl/cu113

# Install other dependencies
CC=/usr/bin/gcc-9 CXX=/usr/bin/g++-9 pip install -r requirements.txt
```

### Environment B (Complex, text2loc_c)

```bash
conda create -n text2loc_c python=3.10 -y
conda activate text2loc_c
CC=/usr/bin/gcc-9 CXX=/usr/bin/g++-9 pip install -r requirements2.txt \
  -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
```

### Environment C (LoRA training, text2loc_lora)

```bash
conda create -n text2loc_lora python=3.11 -y
conda activate text2loc_lora

# Install PyTorch 2.6.0 built for CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# Install other dependencies
CC=/usr/bin/gcc-9 CXX=/usr/bin/g++-9 pip install -r requirements3.txt
```

## Dataset

The data used by this repo comes from **two sources**:

**1. KITTI360Pose** — the point-cloud cells and pose annotations under
`data/k360_30-10_scG_pd10_pc4_spY_all/{cells, poses}`. These follow the KITTI360Pose
benchmark introduced by **Text2Pos** (Kolmet et al., CVPR 2022) and used by
**Text2Loc** (Xia et al., CVPR 2024), built on the **KITTI-360** dataset (Liao et al., TPAMI 2022).

**2. Our own data** — everything generated on top of the cells/poses:
- `data/text_pair/text2/` — text-to-text pairs used to train the text2 LoRA (see the Training section).
- `data/k360_.../{texts1, text1_no_color, text2, text2_no_color, text2_train, text2_no_color_train}`
  — our LLM-generated text descriptions (text1 = moderate, text2 = complex; color / no-color;
  `_train` = auxiliary training split).
- `data/pose_dataset_color/`, `data/pose_dataset_no_color/` — our unified gridCell pose datasets,
  built on four public outdoor point-cloud datasets (see below), plus their `texts1 / text2 / text2_train`.

Our pose datasets are built from four public outdoor point-cloud datasets. The scene
`.pkl` / `.json` files under `pose_dataset_color/` and `pose_dataset_no_color/` map to the
source datasets as follows:

| Dataset | Split | Scenes (file names) | Paper |
|---|---|---|---|
| **Paris-CARLA-3D** | color | `Soufflot03`, `Soufflot12`, `Soufflot45` (real, Paris) + `Town01`–`Town07` (synthetic, CARLA) | Deschaud et al., *Paris-CARLA-3D: A Real and Synthetic Outdoor Point Cloud Dataset for Challenging Tasks in 3D Mapping*, 2021 |
| **Toronto-3D** | color | `toronto` | Tan et al., *Toronto-3D: A Large-scale Mobile LiDAR Dataset for Semantic Segmentation of Urban Roadways*, 2020 (vehicle MLS, with RGB, 8 semantic classes) |
| **Paris-Lille-3D** | no_color | `Lille1`, `Lille2`, `Paris` | Roynard et al., *Paris-Lille-3D: A large and high-quality ground-truth urban point cloud dataset for automatic segmentation and classification*, 2017/2018 |
| **TUM-MLS-2016** (TUM City Campus) | no_color | `tum`, `tum_train`, `tum_test` | Zhu et al., *TUM-MLS-2016: An Annotated Mobile LiDAR Dataset of the TUM City Campus for Semantic Point Cloud Interpretation in Urban Areas*, 2020 |

Download the data from:

- [Here: https://huggingface.co/datasets/sltttt/Text2Loc](https://huggingface.co/datasets/sltttt/Text2Loc)

Place it so the final directory structure is:
```
│Text2Loc/
├──dataloading/
├──datapreparation/
├──data/
│   ├──text_pair/
│       ├──text2/
│           ├──text.json
│           ├──text_refine.json
│           ├──text_no_color.json
│           ├──text_no_color_refine.json
│   ├──pose_dataset_color/
│       ├──unified_3_30-10_gridCells_pd10_pc2_shiftPoses_all_nm-6/
│           ├──cells/
│               ├──toronto.pkl
│               ├──...
│           ├──poses/
│               ├──toronto.pkl
│               ├──...
│           ├──direction/
│               ├──toronto.json
│               ├──...
│           ├──texts1/                (text1)
│               ├──toronto.json
│               ├──...
│           ├──text2/                 (text2)
│               ├──toronto.json
│               ├──...
│           ├──text2_train/           (text2 training)
│               ├──toronto.json
│               ├──...
│   ├──pose_dataset_no_color/
│   ├──k360_30-10_scG_pd10_pc4_spY_all/ (KITTI360)
│       ├──cells/
│       ├──poses/
│       ├──texts1/                 
│       ├──text1_no_color/         
│       ├──text2/                  
│       ├──text1_no_color/             
│       ├──text2_train/            
│       ├──text2_no_color_train/   
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

## Inference

The inference code is provided for three language complexity settings via `--text_mode`:
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
conda activate text2loc_c
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

---

# Training

In addition to inference, this repo keeps the **training code**, the **LoRA training scripts**, and the **corresponding launch scripts** for every final model:

```
training/               # 3 training functions
├── coarse.py               # coarse retrieval training (text0)
├── coarse_aux.py           # coarse retrieval training (text1 / text2, with auxiliary text)
└── fine.py                 # fine localization training (all text modes)
training_lora/          # 4 LoRA training scripts (text2 only)
├── training_lora2.py               → lora/t5-large               (color-baseline)
├── training_lora2_fine.py          → lora/t5-large_fine          (color-refine)
├── training_lora2_no_color.py      → lora/t5-large_no_color          (nocolor-baseline)
└── training_lora2_no_color_fine.py → lora/t5-large_no_color_fine     (nocolor-refine)
scripts/                # launch scripts (named text{x}_{color|nocolor}_{baseline|refine})
├── coarse/text{0,1,2}_{color,nocolor}_{baseline,refine}.sbatch   (12)
├── fine/  text{0,1,2}_{color,nocolor}_{baseline,refine}.sbatch   (12)
└── lora/  text2_{color,nocolor}_{baseline,refine}.sbatch         (4, text2 only)
```

Each script activates the matching environment: **text0 / text1 → `text2loc_sm`**; **text2 → `text2loc_c`** (CUDA ≥ 11.7, required for LoRA loading); **lora/ → `text2loc_lora`**.

## Training Pipeline (two stages)

1. **Coarse**: retrieve the top-k candidate cells from all cells.
2. **Fine**: regress the precise position within the retrieved cell.

## Training Functions (training/)

| text mode | Coarse training function | Notes |
|---|---|---|
| **text0** | `python -m training.coarse` | no auxiliary text; main encoder t5-small |
| **text1** | `python -m training.coarse_aux` | auxiliary-text branch; aux encoder t5-large (**no LoRA**) |
| **text2** | `python -m training.coarse_aux` | auxiliary-text branch; aux encoder t5-large **+ LoRA** |

- **Fine (all text modes)**: `python -m training.fine`.

## LoRA Training (training_lora/) — text2 only

The text2 auxiliary text encoder (t5-large) is first fine-tuned with LoRA (a seq2seq task: input text → output text). The 4 versions are trained on different datasets:

| Script | Training data | Output LoRA | Used by |
|---|---|---|---|
| `training_lora2.py` | text.json | `lora/t5-large` | color-baseline_text2 |
| `training_lora2_fine.py` | text_refine.json | `lora/t5-large_refine` | color-refine_text2 |
| `training_lora2_no_color.py` | text_no_color.json | `lora/t5-large_no_color` | nocolor-baseline_text2 |
| `training_lora2_no_color_fine.py` | text_no_color_refine.json | `lora/t5-large_no_color_refine` | nocolor-refine_text2 |

> `_fine` = refine/finetune (unified gridCells) dataset; no suffix = baseline (k360) dataset.

## Full Training Map (4 versions)

Each "version" = {color | nocolor} × {baseline | refine}, covering coarse + fine for text0/1/2.

| Version | text0/1/2 coarse | fine (all) | text2 aux LoRA script |
|---|---|---|---|
| color-baseline | coarse.py / coarse_aux.py / coarse_aux.py | fine.py | training_lora2.py |
| color-refine | coarse.py / coarse_aux.py / coarse_aux.py | fine.py | training_lora2_fine.py |
| nocolor-baseline | coarse.py / coarse_aux.py / coarse_aux.py | fine.py | training_lora2_no_color.py |
| nocolor-refine | coarse.py / coarse_aux.py / coarse_aux.py | fine.py | training_lora2_no_color_fine.py |

Launch scripts live in `scripts/coarse/`, `scripts/fine/`, and `scripts/lora/` (file names are `text{x}_{color|nocolor}_{baseline|refine}.sbatch`).

<!-- ## ⚠️ Important: text2 (training / evaluation / inference) requires CUDA ≥ 11.7

`models/language_encoder.py` loads the t5-large LoRA only when `float(torch.version.cuda) >= 11.7`. Under CUDA < 11.7 the **LoRA is silently skipped** (the aux encoder falls back to the un-finetuned base t5-large), which markedly lowers text2 results. Run text2 in a **CUDA ≥ 11.7 environment** (`text2loc_c`, torch 2.0.0+cu117); train the LoRA with `text2loc_lora`. text0 / text1 (`text2loc_sm`) do not use LoRA and are unaffected. -->

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{xia2024text2loc,
  title     = {Text2Loc: 3D Point Cloud Localization from Natural Language},
  author    = {Xia, Yan and Shi, Letian and Ding, Zifeng and Henriques, Jo\~{a}o F. and Cremers, Daniel},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages     = {14958--14967},
  year      = {2024}
}

@article{xia2025text2locpp,
  title   = {Text2Loc++: Generalizing 3D Point Cloud Localization from Natural Language},
  author  = {Xia, Yan and Shi, Letian and Di, Yilin and Henriques, Jo\~{a}o F. and Cremers, Daniel},
  journal = {arXiv preprint arXiv:2511.15308},
  year    = {2025}
}
```
