from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from models.language_encoder import LanguageEncoder, get_mlp
from models.object_encoder import ObjectEncoder


class CellRetrievalNetwork(torch.nn.Module):
    def __init__(
        self, known_classes: List[str], known_colors: List[str], args, is_eval = False, 
    ):
        """Coarse module for cell retrieval.
        Implemented as a language encoder and an object encoder.
        The object encoder aggregates information about a varying count of multiple objects through a DGCNN architecture.
        """
        super(CellRetrievalNetwork, self).__init__()
        self.embed_dim = args.coarse_embed_dim

        """
        Object module
        """

        # CARE: possibly handle variation in forward()!

        self.object_encoder = ObjectEncoder(args.coarse_embed_dim, known_classes, known_colors, args, is_eval = is_eval)
        self.object_size = args.object_size
        num_heads = args.object_inter_module_num_heads
        num_layers = args.object_inter_module_num_layers

        self.obj_inter_module = nn.ModuleList([nn.TransformerEncoderLayer(args.coarse_embed_dim, num_heads,  dim_feedforward = 2 * args.coarse_embed_dim) for _ in range(num_layers)])
        self.add_aux_text = args.add_aux_text

        if self.add_aux_text:
            if args.aux_hungging_model != "":
                hungging_model = args.aux_hungging_model
            else:
                hungging_model = args.hungging_model
            self.aux_language_encoder = LanguageEncoder(args.coarse_embed_dim,  
                                                hungging_model = hungging_model, 
                                                fixed_embedding = args.aux_fixed_embedding, 
                                                intra_module_num_layers = args.aux_intra_module_num_layers, 
                                                intra_module_num_heads = args.aux_intra_module_num_heads, 
                                                is_fine = False,  
                                                inter_module_num_layers = args.aux_inter_module_num_layers,
                                                inter_module_num_heads = args.aux_inter_module_num_heads,
                                                inter_pos = args.aux_inter_pos,
                                                intra_proj = args.aux_intra_proj,
                                                inter_proj = args.aux_inter_proj,
                                                intra_pos = args.aux_intra_pos, 
                                                llm_output_layer = args.aux_llm_output_layer,
                                                intra_decoder_num_heads = args.aux_intra_decoder_num_heads, 
                                                intra_decoder_num_layers = args.aux_intra_decoder_num_layers, 
                                                intra_num_queries = args.aux_intra_num_queries,
                                                intra_loss = args.aux_intra_loss, 
                                                intra_mlp = args.aux_intra_mlp, 
                                                split_type= args.aux_split_type, 
                                                mask = args.aux_mask,
                                                version = args.aux_version,
                                                clip_length = args.aux_clip_length,
                                                lora_checkpoint=args.aux_lora_checkpoint,
                                                ) 


        if args.is_global:
            self.mlp_global = get_mlp([2 * args.coarse_embed_dim, args.coarse_embed_dim])

        """
        Textual module
        """

        self.language_encoder = LanguageEncoder(args.coarse_embed_dim,  
                                                hungging_model = args.hungging_model, 
                                                fixed_embedding = args.fixed_embedding, 
                                                intra_module_num_layers = args.intra_module_num_layers, 
                                                intra_module_num_heads = args.intra_module_num_heads, 
                                                is_fine = False,  
                                                inter_module_num_layers = args.inter_module_num_layers,
                                                inter_module_num_heads = args.inter_module_num_heads,
                                                inter_pos = args.inter_pos,
                                                intra_proj = args.intra_proj,
                                                inter_proj = args.inter_proj,
                                                intra_mlp = args.intra_mlp,
                                                split_type= args.split_type, 
                                                mask = args.mask,
                                                version = args.version,
                                                clip_length = args.clip_length,
                                                lora_checkpoint=args.lora_checkpoint,
                                                ) 


        print(
            f"CellRetrievalNetwork, class embed {args.class_embed}, color embed {args.color_embed}, dim: {args.coarse_embed_dim}, features: {args.use_features}"
        )

        
        if args.add_anchor:
            self.add_anchor = True
            self.pos_anchor_size = args.anchor_size
            x_pos = np.linspace(0,1,self.pos_anchor_size + 1)
            y_pos = np.linspace(0,1,self.pos_anchor_size + 1)
            x_pos = (x_pos[:self.pos_anchor_size] + x_pos[1:]) / 2
            y_pos = (y_pos[:self.pos_anchor_size] + y_pos[1:]) / 2
            xx_pos, yy_pos = np.meshgrid(x_pos, y_pos)
            self.pos_anchor = np.stack([xx_pos.flatten(), yy_pos.flatten()], axis= -1)
            self.mlp_pos = get_mlp([2, 36, args.coarse_embed_dim])
            self.mlp_pos2 =   get_mlp([2 * args.coarse_embed_dim, args.coarse_embed_dim])    


    def encode_text(self, descriptions, length = [], hints_number = [], inst_list = [], fold = 0):

        description_encodings, logits_dict, inst_encodings = self.language_encoder(descriptions, length, hints_number, inst_list = inst_list, fold = fold)  # [B, DIM]

        description_encodings = F.normalize(description_encodings)
        if inst_encodings is not None:
            inst_encodings = F.normalize(inst_encodings)

        return description_encodings, logits_dict, inst_encodings
    
    def encode_aux_text(self, descriptions, length = [], hints_number = [], inst_list = [], fold = 0):

        description_encodings, logits_dict, inst_encodings = self.aux_language_encoder(descriptions, length, hints_number, inst_list = inst_list, fold = fold)  # [B, DIM]

        description_encodings = F.normalize(description_encodings)
        if inst_encodings is not None:
            inst_encodings = F.normalize(inst_encodings)

        return description_encodings, logits_dict, inst_encodings

    def encode_objects(self, objects, object_points, global_points=None, is_instance = False, inst_list = [], object_dropout=0.0, submap_dropout=0.0):
        """
        Process the objects in a flattened way to allow for the processing of batches with uneven sample counts
        """
        
        batch = []  # Batch tensor to send into PyG
        for i_batch, objects_sample in enumerate(objects):
            for obj in objects_sample:
                # class_idx = self.known_classes.get(obj.label, 0)
                # class_indices.append(class_idx)
                batch.append(i_batch)
        batch = torch.tensor(batch, dtype=torch.long, device=self.device)

        # TODO: Norm embeddings or not?
        embeddings, pos_postions, global_embeddings, class_pred = self.object_encoder(objects, object_points, global_points)

        if object_dropout > 0.0:
            mask = (torch.rand(embeddings.shape, device=embeddings.device) > object_dropout).float().to(self.device)
            embeddings = embeddings * mask / (1 - object_dropout)

        object_size = self.object_size

        index_list = [0]
        last = 0
        
        for obj in objects:
            index_list.append(last + len(obj))
            last += len(obj)
        
        embeddings = F.normalize(embeddings, dim=-1) 

        if object_size != -1:
            
            x = torch.zeros(len(objects), object_size, self.embed_dim).to(self.device)
            padding_mask = None

            if inst_list != [] and inst_list[0] != None:
                N, L, C = x.shape
                L_I = len(inst_list[0])
                inst_encodings = torch.zeros(N, L_I, C).to(self.device) 

            else:
                inst_encodings = None

            for idx in range(len(index_list) - 1):
                num_object_raw = index_list[idx + 1] - index_list[idx]
                start = index_list[idx]
                num_object = num_object_raw if num_object_raw <= object_size else object_size
                x[idx,: num_object] = embeddings[start : (start + num_object)]
                
                if inst_encodings is not None:

                    for i_inst, instance_idx in enumerate(inst_list[idx]):
                        inst_encodings[idx, i_inst] = embeddings[start + instance_idx]

        else:

            max_length = max([len(obj) for obj in objects])
            x = torch.zeros(len(objects), max_length, self.embed_dim).to(self.device)
            padding_mask = torch.ones(len(objects), max_length).to(self.device)

            if inst_list != [] and inst_list[0] != None:
                N, L, C = x.shape
                L_I = len(inst_list[0])
                inst_encodings = torch.zeros(N, L_I, C).to(self.device) 

            else:
                inst_encodings = None

            for idx in range(len(index_list) - 1):
                num_object_raw = index_list[idx + 1] - index_list[idx]
                start = index_list[idx]
                num_object = num_object_raw
                x[idx,: num_object] = embeddings[start : (start + num_object)]
                padding_mask[idx, :num_object] = 0
                
                if inst_encodings is not None:

                    for i_inst, instance_idx in enumerate(inst_list[idx]):
                        inst_encodings[idx, i_inst] = embeddings[start + instance_idx]


        if inst_encodings is not None:         
            inst_encodings = inst_encodings.view(N * L_I, C)

        # print(inst_list)
  
        x = x.permute(1, 0, 2).contiguous()
        for idx in range(len(self.obj_inter_module)):
            x = self.obj_inter_module[idx](x, src_key_padding_mask =  padding_mask.bool() if padding_mask is not None else None)

        del embeddings, pos_postions
        
        x = x.max(dim = 0)[0]
        
        if global_embeddings is not None:
            x = self.mlp_global(torch.cat([x, global_embeddings], dim=-1))

        if hasattr(self, "add_anchor"):
            anchor = torch.FloatTensor(self.pos_anchor).to(self.device)
            anchor_embedding = self.mlp_pos(anchor)
            anchor_embedding = anchor_embedding.unsqueeze(0).repeat(x.shape[0], 1, 1)
            x = x.unsqueeze(1).repeat(1, anchor_embedding.shape[1], 1)
            x = torch.cat([x, anchor_embedding], dim = -1)
            N, N_pos, _ = x.shape
            x = x.view(N * N_pos, -1)
            x = self.mlp_pos2(x)
            x = x.view(N, N_pos, -1)
        else:
            anchor = None

        if submap_dropout > 0.0:
            mask = (torch.rand(x.shape, device=x.device) > submap_dropout).float().to(self.device)
            x = x * mask / (1 - submap_dropout)

        
        x = F.normalize(x)

        return x, class_pred, anchor, inst_encodings

    def forward(self):
        raise Exception("Not implemented.")

    @property
    def device(self):
        return self.language_encoder.device

    def get_device(self):
        return self.language_encoder.device

