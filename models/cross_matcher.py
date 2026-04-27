from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import time

from models.language_encoder import get_mlp, LanguageEncoder
from models.object_encoder import ObjectEncoder


from datapreparation.kitti360pose.imports import Object3d as Object3d_K360


def get_mlp_offset(dims: List[int], add_batchnorm=False) -> nn.Sequential:
    """Return an MLP without trailing ReLU or BatchNorm for Offset/Translation regression.

    Args:
        dims (List[int]): List of dimension sizes
        add_batchnorm (bool, optional): Whether to add a BatchNorm. Defaults to False.

    Returns:
        nn.Sequential: Result MLP
    """
    if len(dims) < 3:
        print("get_mlp(): less than 2 layers!")
    mlp = []
    for i in range(len(dims) - 1):
        mlp.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            mlp.append(nn.ReLU())
            if add_batchnorm:
                mlp.append(nn.BatchNorm1d(dims[i + 1]))
    return nn.Sequential(*mlp)


class CrossMatch(torch.nn.Module):
    def __init__(
        self, known_classes: List[str], known_colors: List[str], args
    ):
        """Fine hints-to-objects matching module.
        Consists of object-encoder, language-encoder and SuperGlue-based matching module.

        Args:
            known_classes (List[str]): List of known classes
            known_colors (List[str]): List of known colors
            args: Global training args
        """
        super(CrossMatch, self).__init__()
        self.embed_dim = args.fine_embed_dim

        self.object_encoder = ObjectEncoder(args.fine_embed_dim, known_classes, known_colors, args)

        if args.fine_hungging_model != "":
            hungging_model = args.fine_hungging_model
        else:
            hungging_model = args.hungging_model

        self.language_encoder = LanguageEncoder(args.fine_embed_dim,  
                                                hungging_model = hungging_model, 
                                                fixed_embedding = args.fixed_embedding, 
                                                intra_module_num_layers = args.fine_intra_module_num_layers, 
                                                intra_module_num_heads = args.fine_intra_module_num_heads, 
                                                is_fine = True, 
                                                inter_pos = args.inter_pos,
                                                intra_proj = args.intra_proj,
                                                intra_mlp = args.intra_mlp,
                                                split_type= args.split_type, 
                                                mask = args.mask,
                                                version = args.version,
                                                clip_length = args.clip_length,
                                                lora_checkpoint=args.fine_lora_checkpoint,
                                                )

        if False:
            if args.fine_aux_hungging_model != "":
                hungging_model = args.fine_aux_hungging_model
            elif args.aux_hungging_model != "":
                hungging_model = args.aux_hungging_model
            else:
                hungging_model = args.hungging_model
            self.aux_language_encoder = LanguageEncoder(args.fine_embed_dim,  
                                        hungging_model = hungging_model, 
                                        fixed_embedding = args.fixed_embedding, 
                                        intra_module_num_layers = args.fine_intra_module_num_layers, 
                                        intra_module_num_heads = args.fine_intra_module_num_heads, 
                                        is_fine = True, 
                                        inter_pos = args.inter_pos,
                                        intra_proj = args.intra_proj,
                                        intra_mlp = args.intra_mlp,
                                        split_type= args.split_type, 
                                        mask = args.mask,
                                        version = args.version,
                                        clip_length = args.clip_length,
                                        )
        else:
            self.aux_language_encoder = None

        
        self.mlp_offsets = get_mlp_offset([self.embed_dim, self.embed_dim // 2, 2])
        self.text_mode = args.text_mode

        if args.fine_num_decoder_layers > 0:
            self.cross_hints = nn.ModuleList([nn.TransformerDecoderLayer(d_model = args.fine_embed_dim, 
                                                    nhead = args.fine_num_decoder_heads, 
                                                    dim_feedforward = args.fine_embed_dim * 4) for _ in range(args.fine_num_decoder_layers)])

            self.cross_objects = nn.ModuleList([nn.TransformerDecoderLayer(d_model = args.fine_embed_dim, 
                                                    nhead = args.fine_num_decoder_heads, 
                                                    dim_feedforward = args.fine_embed_dim * 4) for _ in range(args.fine_num_decoder_layers)])
        else:
            self.cross_hints = nn.TransformerDecoderLayer(d_model = args.fine_embed_dim, 
                                                    nhead = args.fine_num_decoder_heads, 
                                                    dim_feedforward = args.fine_embed_dim * 4)
            self.cross_objects = None

        

    def forward(self, objects, hints, object_points, sub_hints = None, text_only = False, branch_type = 0, time_sum = None):
        batch_size = len(objects)
        num_objects = len(objects[0])

        """
        Text encoder
        """
        if hasattr(self, "text_mode"):
            use_length = self.text_mode != 0
        else:
            use_length = False

        if time_sum is not None:
            start = time.time()
        
        if text_only:
            if branch_type == 1:
                if sub_hints is not None:
                    sub_out_text = self.aux_language_encoder(sub_hints, use_length = use_length)
                else:
                    sub_out_text = None
                out_text = self.aux_language_encoder(hints, use_length = use_length)
            else:
                if sub_hints is not None:
                    sub_out_text = self.language_encoder(sub_hints, use_length = use_length)
                else:
                    sub_out_text = None
                out_text = None
            if sub_out_text is not None and type(sub_out_text) is tuple:
                sub_hint_encodings = sub_out_text[0]
            else:
                sub_hint_encodings = None
        else: 
            out_text = self.language_encoder(hints, use_length = use_length)
            sub_hint_encodings = None

        if time_sum is not None:
            end = time.time()
            time_sum += end - start

        if type(out_text) is tuple:
            hint_encodings = out_text[0]
            padding_mask = out_text[2]
        else:
            hint_encodings = out_text
            padding_mask = None
        
        if hint_encodings is not None:
            hint_encodings = F.normalize(hint_encodings, dim=-1)

        if text_only and sub_hint_encodings is not None:
            out_hint_encodings = sub_hint_encodings.max(1)[0]    
            out_hint_encodings = F.normalize(out_hint_encodings, dim=-1)
        else:
            out_hint_encodings = None
        
        if text_only and branch_type == 0:
            return None, out_hint_encodings

        """
        Object encoder
        """
        out = self.object_encoder(objects, object_points)
        if type(out) is tuple:
            object_encodings = out[0]
            pos_postions = out[1]
        else:
            object_encodings = out

        object_encodings = object_encodings.reshape((batch_size, num_objects, self.embed_dim))
        object_encodings = F.normalize(object_encodings, dim=-1)

        """
        Match object-encodings to hint-encodings
        """
        desc0 = object_encodings.transpose(0, 1)  # [num_obj, B, DIM]

        if time_sum is not None:
            start = time.time()

        desc1 = hint_encodings.transpose(0, 1)  # [num_hints, B, DIM]

        if self.cross_objects is not None:
            if len(self.cross_hints) == len(self.cross_objects):
                for i in range(len(self.cross_hints)):
                    desc0 = self.cross_objects[i](desc0, desc1, memory_key_padding_mask = padding_mask.bool() if padding_mask is not None else None)
                    desc1 = self.cross_hints[i](desc1, desc0, tgt_key_padding_mask = padding_mask.bool() if padding_mask is not None else None)
            else:
                desc0_new = self.cross_objects[0](desc0, desc1, memory_key_padding_mask = padding_mask.bool() if padding_mask is not None else None)
                desc1_new = self.cross_hints[0](desc1, desc0, tgt_key_padding_mask = padding_mask.bool() if padding_mask is not None else None)
                desc1 = self.cross_hints[1](desc1_new, desc0_new, tgt_key_padding_mask = padding_mask.bool() if padding_mask is not None else None)
        else:
            desc1 = self.cross_hints(desc1, desc0, tgt_key_padding_mask = padding_mask.bool() if padding_mask is not None else None)

        desc1 = desc1.max(dim=0)[0]
        offsets = self.mlp_offsets(desc1)

        if time_sum is not None:
            end = time.time()
            time_sum += end - start
        
        if out_hint_encodings is not None:
            return offsets, out_hint_encodings
        else:
            return offsets, time_sum

    @property
    def device(self):
        return next(self.mlp_offsets.parameters()).device
    def get_device(self):
        return next(self.mlp_offsets.parameters()).device


def get_pos_in_cell(objects: List[Object3d_K360], matches0, offsets):
    """Extract a pose estimation relative to the cell (∈ [0,1]²) by
    adding up for each matched objects its location plus offset-vector of corresponding hint,
    then taking the average.

    Args:
        objects (List[Object3d_K360]): List of objects of the cell
        matches0 : matches0 from SuperGlue
        offsets : Offset predictions for each hint

    Returns:
        np.ndarray: Pose estimate
    """
    pose_preds = []  # For each match the object-location plus corresponding offset-vector
    for obj_idx, hint_idx in enumerate(matches0):
        if obj_idx == -1 or hint_idx == -1:
            continue
        # pose_preds.append(objects[obj_idx].closest_point[0:2] + offsets[hint_idx]) # Object location plus offset of corresponding hint
        pose_preds.append(
            objects[obj_idx].get_center()[0:2] + offsets[hint_idx]
        )  # Object location plus offset of corresponding hint
    return (
        np.mean(pose_preds, axis=0) if len(pose_preds) > 0 else np.array((0.5, 0.5))
    )  # Guess the middle if no matches


def intersect(P0, P1):
    n = (P1 - P0) / np.linalg.norm(P1 - P0, axis=1)[:, np.newaxis]  # normalized
    projs = np.eye(n.shape[1]) - n[:, :, np.newaxis] * n[:, np.newaxis]  # I - n*n.T
    R = projs.sum(axis=0)
    q = (projs @ P0[:, :, np.newaxis]).sum(axis=0)
    p = np.linalg.lstsq(R, q, rcond=None)[0]
    return p


def get_pos_in_cell_intersect(objects: List[Object3d_K360], matches0, directions):
    directions /= np.linalg.norm(directions, axis=1)[:, np.newaxis]
    points0 = []
    points1 = []
    for obj_idx, hint_idx in enumerate(matches0):
        if obj_idx == -1 or hint_idx == -1:
            continue
        points0.append(objects[obj_idx].get_center()[0:2])
        points1.append(objects[obj_idx].get_center()[0:2] + directions[hint_idx])
    if len(points0) < 2:
        return np.array((0.5, 0.5))
    else:
        return intersect(np.array(points0), np.array(points1))

