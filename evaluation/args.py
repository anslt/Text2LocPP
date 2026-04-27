import argparse
from argparse import ArgumentParser
import os
import os.path as osp


def parse_arguments():
    parser = argparse.ArgumentParser(description="Text2Loc Evaluation")

    # Paths
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dataset", type=str, default="K360")
    parser.add_argument("--base_path", type=str)
    parser.add_argument("--path_coarse", type=str, help="The path to the Cell-Retrieval model")
    parser.add_argument(
        "--path_fine", type=str, help="The path to the Hints-to-Objects matching model"
    )

    # Options
    parser.add_argument("--top_k", type=int, nargs="+", default=[1, 3, 5, 10])
    parser.add_argument(
        "--threshs", type=int, nargs="+", default=[5, 10, 15]
    )  # Possibly over-write here when it becomes a list of tuples
    parser.add_argument("--use_features", nargs="+", default=["class", "color", "position", "num"])
    parser.add_argument(
        "--use_test_set", action="store_true", help="Run test-set instead of the validation set."
    )

    parser.add_argument("--no_pc_augment", action="store_true")
    parser.add_argument("--no_pc_augment_fine", action="store_true")

    parser.add_argument(
        "--plot_retrievals",
        action="store_true",
        help="Plot 3 success and fail examples, then quit.",
    )
    parser.add_argument("--plot_matches", action="store_true")
    parser.add_argument("--coarse_only", action="store_true")
    parser.add_argument("--fine_only", action="store_true")

    # Fine
    parser.add_argument("--fine_embed_dim", type=int, default=128)
    parser.add_argument("--fine_num_decoder_heads", type=int, default=4)
    parser.add_argument("--fine_num_decoder_layers", type=int, default=2)

    parser.add_argument("--pad_size", type=int, default=16)
    parser.add_argument("--num_mentioned", type=int, default=6)
    parser.add_argument(
        "--regenerate_coarse_descriptions",
        action="store_true",
        help=(
            "Re-build hint text from GT pose_w and database cell (describe_best_cell-style), "
            "using --num_mentioned and each pose's described_by (fallback: closest)."
        ),
    )
    parser.add_argument("--describe_by", type=str, default="all")

    # Loss
    parser.add_argument("--margin", type=float, default=0.35)  # Before: 0.5
    parser.add_argument("--temperature", type=float, default=0.1) 
    parser.add_argument("--ranking_loss", type=str, default="pairwise")

    # Object-encoder / PointNet
    parser.add_argument("--coarse_embed_dim", type=int, default=256)
    parser.add_argument("--pointnet_layers", type=int, default=3)
    parser.add_argument("--pointnet_variation", type=int, default=0)
    parser.add_argument("--pointnet_numpoints", type=int, default=256)
    parser.add_argument(
        "--pointnet_path", type=str, default="./checkpoints/pointnet_acc0.86_lr1_p256.pth"
    )
    parser.add_argument("--pointnet_freeze", action="store_true")
    parser.add_argument("--pointnet_features", type=int, default=2)

    parser.add_argument("--class_embed", action="store_true")
    parser.add_argument("--color_embed", action="store_true")
    
    parser.add_argument("--object_size", type=int, default=28)
    parser.add_argument("--object_inter_module_num_heads", type=int, default=4)
    parser.add_argument("--object_inter_module_num_layers", type=int, default=2)

    # Language Encoder
    parser.add_argument("--hungging_model", type=str, default="t5-small", help="hugging face model")
    parser.add_argument("--fine_hungging_model", type=str, default="", help="hugging face model")
    parser.add_argument("--fixed_embedding", action="store_true")

    parser.add_argument("--inter_module_num_heads", type=int, default=4)
    parser.add_argument("--inter_module_num_layers", type=int, default=1)
    parser.add_argument("--intra_module_num_heads", type=int, default=4)
    parser.add_argument("--intra_module_num_layers", type=int, default=1)
    parser.add_argument("--fine_intra_module_num_heads", type=int, default=4)
    parser.add_argument("--fine_intra_module_num_layers", type=int, default=1)

    parser.add_argument("--inter_pos", action="store_true")
    parser.add_argument("--intra_proj", type=str, default="max")
    parser.add_argument("--inter_proj", type=str, default="max")
    parser.add_argument("--intra_pos", action="store_true")

    parser.add_argument("--llm_output_layer", type=float, default=-1)
    parser.add_argument("--intra_decoder_num_heads", type=int, default=4)
    parser.add_argument("--intra_decoder_num_layers", type=int, default=0)
    parser.add_argument("--intra_num_queries", type=int, default=10)
    parser.add_argument("--intra_loss", action="store_true")
    parser.add_argument("--intra_mlp", action="store_true")
    parser.add_argument("--split_type", type=str, default="sentence")
    parser.add_argument("--mask", action="store_true")
    parser.add_argument("--version", type=str, default="old")
    parser.add_argument("--clip_length", action="store_true")
    parser.add_argument("--lora_checkpoint", type=str, default="")
    parser.add_argument("--fine_lora_checkpoint", type=str, default="")

    parser.add_argument("--aux_hungging_model", type=str, default="", help="hugging face model")
    parser.add_argument("--fine_aux_hungging_model", type=str, default="", help="hugging face model")
    parser.add_argument("--aux_fixed_embedding", action="store_true")

    parser.add_argument("--aux_inter_module_num_heads", type=int, default=4)
    parser.add_argument("--aux_inter_module_num_layers", type=int, default=1)
    parser.add_argument("--aux_intra_module_num_heads", type=int, default=4)
    parser.add_argument("--aux_intra_module_num_layers", type=int, default=1)

    parser.add_argument("--aux_inter_pos", action="store_true")
    parser.add_argument("--aux_intra_proj", type=str, default="max")
    parser.add_argument("--aux_inter_proj", type=str, default="max")
    parser.add_argument("--aux_intra_pos", action="store_true")

    parser.add_argument("--aux_llm_output_layer", type=float, default=-1)
    parser.add_argument("--aux_intra_decoder_num_heads", type=int, default=4)
    parser.add_argument("--aux_intra_decoder_num_layers", type=int, default=0)
    parser.add_argument("--aux_intra_num_queries", type=int, default=10)
    parser.add_argument("--aux_intra_loss", action="store_true")
    parser.add_argument("--aux_intra_mlp", action="store_true")
    parser.add_argument("--aux_split_type", type=str, default="sentence")
    parser.add_argument("--aux_mask", action="store_true")
    parser.add_argument("--aux_version", type=str, default="old")
    parser.add_argument("--aux_clip_length", action="store_true")
    parser.add_argument("--aux_lora_checkpoint", type=str, default="") 

    parser.add_argument("--hint_type", type=int, default=0)
    parser.add_argument("--no_color0", action="store_true")
    parser.add_argument("--no_color", action="store_true") # in training, no color in text, in pretraining no color constraint
    parser.add_argument("--no_color2", action="store_true") # no color in object and in object encoder
    parser.add_argument("--configs", type=str, default="kitti360")
    parser.add_argument("--use_all_set", action="store_true")
    parser.add_argument("--is_global", action="store_true")
    parser.add_argument("--text_mode", type=int, default=0)

    parser.add_argument("--add_anchor", action="store_true")
    parser.add_argument("--add_aux_text", action="store_true")

    parser.add_argument("--change_type", type=str, default="none")
    parser.add_argument("--change_num", type=int, default=0)
    parser.add_argument("--cell_multiple", type=int, default=0)
    
    parser.add_argument("--fix_image", action="store_true")
    parser.add_argument("--image_version", type=str, default="resnet18")

    parser.add_argument("--opposite", action="store_true")

    args = parser.parse_args()

    assert osp.isfile(args.path_coarse)
    if args.path_fine:
        assert osp.isfile(args.path_fine)

    return args

