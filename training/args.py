import argparse
from argparse import ArgumentParser
import os.path as osp


def parse_arguments():
    parser = argparse.ArgumentParser(description="Text2Loc Training")

    # General

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_batches", type=int, default=None)
    parser.add_argument("--dataset", type=str, default="K360", help="Currently only K360")
    parser.add_argument("--base_path", type=str, help="Root path of Kitti360Pose")

    # Model
    parser.add_argument("--use_features", nargs="+", default=["class", "color", "position", "num"])
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--learning_rate", default=0.001, type=float, help="Learning rate")

    parser.add_argument(
        "--continue_path", type=str, help="Set to continue from a previous checkpoint"
    )

    parser.add_argument("--no_pc_augment", action="store_true")

    # Fine
    parser.add_argument("--fine_embed_dim", type=int, default=128)
    parser.add_argument("--offset_lambda", type=float, default=5)
    parser.add_argument("--pmc_prob", type=float, default=0.0)
    parser.add_argument("--pmc_threshold", type=float, default=0.4)
    parser.add_argument("--fine_shuffle", action="store_true")
    parser.add_argument("--gt_area", action="store_true")
    parser.add_argument("--index_area", type=int, default=0)

    parser.add_argument("--fine_num_decoder_heads", type=int, default=4)
    parser.add_argument("--fine_num_decoder_layers", type=int, default=2)

    parser.add_argument("--pad_size", type=int, default=16)
    parser.add_argument("--num_mentioned", type=int, default=6)
    parser.add_argument("--describe_by", type=str, default="all")

    # Loss
    parser.add_argument("--margin", type=float, default=0.35)  # Before: 0.5
    parser.add_argument("--temperature", type=float, default=0.1) 
    parser.add_argument("--temperature_text", type=float, default=0.0) 
    parser.add_argument("--top_k", type=int, nargs="+", default=[1, 3, 5])
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
    parser.add_argument("--clip_length", type=int, default=77)
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
    parser.add_argument("--aux_clip_length", type=int, default=77)
    parser.add_argument("--aux_lora_checkpoint", type=str, default="")

    parser.add_argument("--intra_cost_class",type=float, default=1.0)
    parser.add_argument("--intra_cost_color",type=float, default=1.0)
    parser.add_argument("--intra_cost_direction",type=float, default=1.0)
    parser.add_argument("--intra_eos_coef",type=float, default=0.5)
    


    # Variations which tranlations are fed into the network for training/evaluation.
    # NOTE: These variations did not make much difference.
    parser.add_argument("--regressor_cell", type=str, default="pose")  # Pose or best
    parser.add_argument("--regressor_learn", type=str, default="center")  # Center or closest
    parser.add_argument("--regressor_eval", type=str, default="center")  # Center or closest

    # Others
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--lr_gamma", type=float, default=1.0)
    parser.add_argument("--lr_scheduler", type=str, default="exponential")
    parser.add_argument("--lr_step", type=float, default=10)
    parser.add_argument("--folder_name", type=str, default="folder_name")

    parser.add_argument("--cpus", type=int, default=0)
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--hint_type", type=int, default=0)
    parser.add_argument("--add_object_number", action="store_true")
    parser.add_argument("--random_choice_direction", action="store_true")
    parser.add_argument("--replace_pose", action="store_true")

    parser.add_argument("--no_color0", action="store_true") # control the color setting in dataset
    parser.add_argument("--no_color", action="store_true") # in training, no color in text, in pretraining no color constraint
    parser.add_argument("--replace_color", action="store_true") # in training, replace color by semantic
    parser.add_argument("--replace_random", action="store_true") 
    parser.add_argument("--no_color2", action="store_true") # no color in object and in object encoder replace color by xyz
    parser.add_argument("--pre_color_coeff", type=float, default=0.1)
    parser.add_argument("--configs", type=str, default="kitti360")
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--is_global", action="store_true") # add globel module
    parser.add_argument("--constraint_all", action="store_true")
    parser.add_argument("--random_discard", action="store_true")
    parser.add_argument("--gt_random_shift", type=float, default=0.0, help="Max random shift (normalized) around GT, e.g. 0.05 = 5%% of cell size")
    parser.add_argument("--perfect_matching", action="store_true")

    parser.add_argument("--corrupt_prob", type=float, default=0.0, help="Prob of corrupting text per sample during training")
    parser.add_argument("--corrupt_num", type=int, default=1, help="Number of sentences to corrupt per sample")
    parser.add_argument("--corrupt_coeff", type=float, default=0.5, help="Loss weight for corrupted text contrastive loss")

    parser.add_argument("--add_anchor", action="store_true")
    parser.add_argument("--combine_anchor_loss", action="store_true")
    parser.add_argument("--anchor_size", type=int, default=6)
    parser.add_argument("--anchor_coeff", type=float, default=0.1)
    parser.add_argument("--pos_threshold", type=float, default=0.333333333333333)
    parser.add_argument("--pos_net", action="store_true") # no use
    # parser.add_argument("--anchor_freq", type=int, default=5)

    parser.add_argument("--weight_loss", action="store_true")
    parser.add_argument("--text_mode", type=int, default=0)
    parser.add_argument("--sub_sample", action="store_true")

    parser.add_argument("--text_loss", action="store_true")
    parser.add_argument("--text_loss_coeff", type=float, default=1.0)
    parser.add_argument("--add_aux_text", action="store_true")
    parser.add_argument("--transfer_mode", action="store_true")
    
    parser.add_argument("--aux_prepare", action="store_true")
    parser.add_argument("--aux_epochs", type=int, default=2)
    parser.add_argument("--aux_limit", type=int, default=2)
    parser.add_argument("--aux_epochs2", type=int, default=2)
    parser.add_argument("--memory_efficient", action="store_true")

    parser.add_argument("--inst_num", type=int, default=0)
    parser.add_argument("--loss_inst_coeff", type=float, default=0.0)
    parser.add_argument("--pre_inst_epochs", type=int, default=0)

    parser.add_argument("--fold", type=int, default=0)

    parser.add_argument("--fix_lang", action="store_true")

    parser.add_argument("--fix_image", action="store_true")
    parser.add_argument("--image_version", type=str, default="resnet18")


    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--no_remove", action="store_true")
    parser.add_argument("--random_generate", action="store_true")

    parser.add_argument("--object_dropout", type=float, default=0.0)
    parser.add_argument("--submap_dropout", type=float, default=0.0)

    parser.add_argument("--subset", type=int, default=0)
    parser.add_argument("--org_cell", action="store_true")
    parser.add_argument("--org_cell_coeff", type=float, default=1.0)
    parser.add_argument("--org_text", action="store_true")
    parser.add_argument("--org_text_coeff", type=float, default=1.0)


    parser.add_argument(
        "--finetune_path", type=str, default=""
    )

    args = parser.parse_args()

    if bool(args.continue_path):
        assert osp.isfile(args.continue_path)

    assert args.regressor_cell in ("pose", "best", "all")
    assert args.regressor_learn in ("center", "closest")
    assert args.regressor_eval in ("center", "closest")

    args.dataset = args.dataset.upper()
    assert args.dataset in ("S3D", "K360")

    assert args.ranking_loss in ("triplet", "pairwise", "hardest", "contrastive", "contrastive2", "contrastive_new", "cosine")

    for feat in args.use_features:
        assert feat in ["class", "color", "position", "num", "image"], "Unexpected feature"

    if args.pointnet_path:
        assert osp.isfile(args.pointnet_path)

    assert osp.isdir(args.base_path)

    assert args.describe_by in ("closest", "class", "direction", "random", "random_direction", "all")

    assert args.text_mode in (0, 1, 2, 3)

    return args


if __name__ == "__main__":
    args = parse_arguments()
    print(args)
