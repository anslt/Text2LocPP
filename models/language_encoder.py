from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from configs.mixed0 import CLASS_TO_INDEX, COLOR_TO_INDEX, DIRECTION_TO_INDEX
#from Long_CLIP.model import longclip

from nltk import tokenize as text_tokenize
# if nltk not work well add the following command
# nltk.download('punkt')
from transformers import AutoTokenizer, T5EncoderModel, LlamaModel, AutoModelForSeq2SeqLM
import clip

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=50):
        super().__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


# CARE: This has a trailing ReLU!!
def get_mlp(channels: List[int], add_batchnorm: bool = True) -> nn.Sequential:
    """Construct and MLP for use in other models.

    Args:
        channels (List[int]): List of number of channels in each layer.
        add_batchnorm (bool, optional): Whether to add BatchNorm after each layer. Defaults to True.

    Returns:
        nn.Sequential: Output MLP
    """
    if add_batchnorm:
        return nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(channels[i - 1], channels[i]), nn.BatchNorm1d(channels[i]), nn.ReLU()
                )
                for i in range(1, len(channels))
            ]
        )
    else:
        return nn.Sequential(
            *[
                nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU())
                for i in range(1, len(channels))
            ]
        )

def get_mlp2(channels: List[int], add_batchnorm: bool = True) -> nn.Sequential:
    """Construct and MLP for use in other models.

    Args:
        channels (List[int]): List of number of channels in each layer.
        add_batchnorm (bool, optional): Whether to add BatchNorm after each layer. Defaults to True.

    Returns:
        nn.Sequential: Output MLP
    """
    if add_batchnorm:
        return nn.Sequential(
            *[
                nn.Sequential(
                    nn.Linear(channels[i - 1], channels[i]), nn.BatchNorm1d(channels[i]), nn.ReLU()
                ) if i < len(channels) - 1
                else
                nn.Sequential(
                    nn.Linear(channels[i - 1], channels[i]), nn.BatchNorm1d(channels[i])
                )
                for i in range(1, len(channels))
            ]
        )
    else:
        return nn.Sequential(
            *[
                nn.Sequential(nn.Linear(channels[i - 1], channels[i]), nn.ReLU())
                if i < len(channels) - 1
                else nn.Sequential(nn.Linear(channels[i - 1], channels[i]))
                for i in range(1, len(channels))
            ]
        )

class LanguageEncoder(torch.nn.Module):
    def __init__(self, embedding_dim,  hungging_model = None, fixed_embedding=False, 
                 intra_module_num_layers=2, intra_module_num_heads=4, 
                 is_fine = False, inter_module_num_layers=2, inter_module_num_heads=4,
                 inter_pos = False, intra_proj = "max", inter_proj = "max", intra_pos = False, llm_output_layer=-1,
                 intra_decoder_num_heads=4, intra_decoder_num_layers=0, intra_num_queries=10,
                 intra_loss=False, intra_mlp=False, split_type="sentence", mask = False, version="old",
                 clip_length=77, lora_checkpoint=""
                 ):
        """Language encoder to encode a set of hints"""
        super(LanguageEncoder, self).__init__()

        self.is_fine = is_fine
        self.intra_proj = intra_proj
        self.inter_proj = inter_proj
        self.intra_loss = intra_loss
        self.mask = mask
        self.version = version
        self.split_type = split_type

        if llm_output_layer == round(llm_output_layer):
            llm_output_layer = int(llm_output_layer)

        assert intra_proj in ["max", "mean", "last"]
        assert inter_proj in ["max", "mean"]
        assert split_type in ["sentence", "all"]
        assert version in ["old", "new"] # compatible to the old version
        if "t5" in hungging_model:

            print("T5")
            print(hungging_model)

            self.llm_tokenizer = AutoTokenizer.from_pretrained(hungging_model)

            use_checkpoints = lora_checkpoint != ""
            if hungging_model == "t5-large" and use_checkpoints and float(torch.version.cuda) >- 11.7:
                AutoModelForSeq2SeqLM._keys_to_ignore_on_load_unexpected = ["decoder.*"]
                self.llm_model = AutoModelForSeq2SeqLM.from_pretrained(lora_checkpoint)
                self.llm_model.decoder = None
                print("checkpoint from LoRA")
                print(lora_checkpoint)
            else:
                T5EncoderModel._keys_to_ignore_on_load_unexpected = ["decoder.*"]
                self.llm_model = T5EncoderModel.from_pretrained(hungging_model)
            self.llm_name = hungging_model
            input_dim = self.llm_model.encoder.embed_tokens.weight.shape[-1]
            assert llm_output_layer == -1
            self.llm_output_layer = llm_output_layer

        elif "llama" in hungging_model:

            print("Llama")
            print(hungging_model)
        
            self.llm_tokenizer = AutoTokenizer.from_pretrained(hungging_model)
            if torch.torch.cuda.device_count() > 1:
                self.llm_model = LlamaModel.from_pretrained(hungging_model, device_map = 'cuda:1')
            else:
                if "8B" not in hungging_model:
                    self.llm_model = LlamaModel.from_pretrained(hungging_model, device_map = 'auto')
                else:
                    self.llm_model = LlamaModel.from_pretrained(hungging_model, torch_dtype=torch.float16, device_map = 'cuda')
            self.llm_name = hungging_model
            self.llm_tokenizer.pad_token = self.llm_tokenizer.eos_token
            self.llm_tokenizer.padding_side = "left"
            self.llm_tokenizer.truncation_side = "left"
            input_dim = self.llm_model.layers[-1].mlp.down_proj.weight.shape[-2]

            if llm_output_layer != -1:
                llm_num_layers = len(self.llm_model.layers)
                if llm_output_layer - round(llm_output_layer) != 0:                   
                    llm_output_layer = int(llm_output_layer * llm_num_layers)
                else:
                    llm_output_layer = int(llm_output_layer)
                assert llm_output_layer < llm_num_layers
                self.llm_output_layer = llm_output_layer
                print("Not output final layer, output layer " + str(llm_output_layer))

            else:
                self.llm_output_layer = llm_output_layer

        elif "longclip" in hungging_model:
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model_version = hungging_model.split("-")[-1]
            self.llm_model, _ = longclip.load(f"./checkpoints/longclip-{model_version }.pt", device=device)
            self.llm_model = self.llm_model.float()
            self.llm_name = hungging_model
            # self.llm_model.visual = None
            input_dim = self.llm_model.token_embedding.embedding_dim
        
        elif "clip" in hungging_model:
            self.llm_name = hungging_model
            load_encoder_folder = "./checkpoints/" 
            saved_encoder, _= clip.load(hungging_model.split("--")[-1], download_root=load_encoder_folder) # example: ViT-B-16
            
            self.llm_tokenize = clip.tokenize
            self.llm_token_embedding = saved_encoder.token_embedding.to(torch.float32)
            self.llm_positional_embedding = saved_encoder.positional_embedding.to(torch.float32)
            self.llm_transformer = saved_encoder.transformer.to(torch.float32)
            # if use_ln_final:
            #     self.ln_final = saved_encoder.ln_final.to(torch.float32)
            # else:
            #     self.ln_final = None # saved_encoder.ln_final.to(torch.float32)
            # self.llm_text_projection = saved_encoder.text_projection.to(torch.float32)
            self.llm_output_layer = llm_output_layer
            self.clip_length = clip_length
            input_dim = self.llm_token_embedding.embedding_dim
        
        else:
            raise NameError("We don't have this LLM.")

        if fixed_embedding:
            self.fixed_embedding = True
            if "clip--" not in hungging_model:
                for para in self.llm_model.parameters():
                    para.requires_grad = False
            else:
                for para in self.llm_transformer.parameters():
                    para.requires_grad = False

        else:
            self.fixed_embedding = False

        hidden_dim = embedding_dim if intra_mlp else input_dim

        if intra_mlp:
            self.intra_mlp = get_mlp2([input_dim, embedding_dim], add_batchnorm=True)
            self.inter_mlp = None

        else:
            
            self.intra_mlp = None
            self.inter_mlp = get_mlp2([input_dim, embedding_dim], add_batchnorm=True)

        if self.version == "old":
            if intra_module_num_layers!= 0:
                self.intra_module = nn.ModuleList([nn.TransformerEncoderLayer(hidden_dim, intra_module_num_heads,  dim_feedforward = input_dim * 4) for _ in range(intra_module_num_layers)])
            else:
                self.intra_module = None
        else:
            encoder_layer = nn.TransformerEncoderLayer(hidden_dim, intra_module_num_heads,  dim_feedforward = input_dim * 4)
            self.intra_module = nn.TransformerEncoder(hidden_dim, intra_module_num_layers)

        if intra_decoder_num_layers != 0:
            if self.version == "old":
                self.intra_decoder = nn.ModuleList([nn.TransformerDecoderLayer(hidden_dim, intra_decoder_num_heads,  dim_feedforward = input_dim * 4) for _ in range(intra_decoder_num_layers)])
            else:
                decoder_layer = nn.TransformerDecoderLayer(hidden_dim, intra_decoder_num_heads,  dim_feedforward = input_dim * 4)
                self.intra_decoder = nn.TransformerDecoder(decoder_layer, intra_decoder_num_layers)
            if intra_loss:
                num_classes = len(CLASS_TO_INDEX) - 1
                num_directions = len(DIRECTION_TO_INDEX)
                num_colors = len(COLOR_TO_INDEX)
                self.class_embed = nn.Linear(hidden_dim, num_classes + 1)
                self.direction_embed = nn.Linear(hidden_dim, num_directions + 1)
                self.color_embed = nn.Linear(hidden_dim, num_colors + 1)
            self.query_embed = nn.Embedding(intra_num_queries, hidden_dim)
        else:
            self.intra_decoder = None

            if intra_pos:
                self.intra_pos_embedding = PositionalEncoding(hidden_dim, max_seq_length=200)
            else:
                self.intra_pos_embedding = None

        
        if not is_fine and split_type == "sentence":
            if self.version == "old":
                if inter_module_num_layers > 0:
                    self.inter_module = nn.ModuleList([nn.TransformerEncoderLayer(embedding_dim, inter_module_num_heads,  dim_feedforward = embedding_dim * 4) for _ in range(inter_module_num_layers)])
                else:
                    self.inter_module = None
            else:
                encoder_layer = nn.TransformerEncoderLayer(embedding_dim, inter_module_num_heads,  dim_feedforward = input_dim * 4)
                self.inter_module = nn.TransformerEncoder(encoder_layer, inter_module_num_layers)

            if inter_pos:
                self.inter_pos_embedding = PositionalEncoding(embedding_dim, max_seq_length=20)
            else:
                self.inter_pos_embedding = None
        else:
            self.inter_module = None
            self.inter_pos_embedding = None

    @torch.no_grad()
    def llama_encoder(self, input_ids, attention_mask = None):

        if self.llm_output_layer == -1:
            llm_out= self.llm_model(input_ids = input_ids, 
                            attention_mask = attention_mask,
                            output_attentions = False)
            
            return llm_out.last_hidden_state
            
        else:
            llm_out = self.llm_model(input_ids = input_ids, 
                            attention_mask = attention_mask,
                            output_attentions = False,
                            output_hidden_states = True)
            
            return llm_out.hidden_states[self.llm_output_layer]
        
    @torch.no_grad()
    def clip_encoder(self, input_ids, length = 77):
        ### NOTE: we don't have dtype function beacause we don't have uniformed model structrue
        ###       This will be fixed in the funture.

        x = self.llm_token_embedding(input_ids) 

        if length == 77:
            positional_embedding = self.llm_positional_embedding
        elif length < 77:
            positional_embedding = self.llm_positional_embedding[:length]
        else:
            raise ValueError
        
        x_16 = x  + positional_embedding 

        x_16 = x_16.permute(1, 0, 2)  # NLD -> LND
        x_16 = self.llm_transformer(x_16)

        x_16 = x_16.permute(1, 0, 2)  # LND -> NLD

        return x_16


    
    def forward(self, descriptions, length = [], hints_number = [], use_length = False, inst_list = [], fold = 0):

        if self.split_type == "sentence":


            split_union_sentences = []

            if length == []:
                length_flag = True
                hints_number = [0] * len(descriptions)
                new_length = []
            else:
                length_flag = False
                new_length = []
            # num_stat = []
            for description in descriptions:
                text_split = text_tokenize.sent_tokenize(description)
                if use_length and length_flag:
                    new_length.append(len(text_split))
                split_union_sentences.extend(text_split)
            
                # num_stat.append(len(text_tokenize.sent_tokenize(description)))

            # if sum(num_stat) != sum(length):
            #     for idx in range(len(length)):
            #         if num_stat[idx] != length[idx]:
            #             print(descriptions[idx])
            #             print(len(text_tokenize.sent_tokenize(descriptions[idx])))
            #             print(length[idx])
            #             import pdb; pdb.set_trace()

            batch_size = len(descriptions)
            num_sentence = len(split_union_sentences) // batch_size if length == [] else max(length)
            if new_length != []:
                num_sentence = max(new_length)

            if "t5" in self.llm_name:
                inputs = self.llm_tokenizer(split_union_sentences, return_tensors="pt", padding = "longest")
                shorten_sentences_indices = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]

                shorten_sentences_indices = shorten_sentences_indices.to(self.device)
                attention_mask = attention_mask.to(self.device)
                with torch.no_grad():
                    
                    if float(torch.version.cuda) >= 11.7:
                        out = self.llm_model.encoder(input_ids = shorten_sentences_indices, 
                                        attention_mask = attention_mask,
                                        output_attentions = False)

                        description_encodings = out[0]
                    else:
                        out = self.llm_model(input_ids = shorten_sentences_indices, 
                                        attention_mask = attention_mask,
                                        output_attentions = False)

                        description_encodings = out.last_hidden_state
            
            elif "llama" in self.llm_name:

                # if extra_model is not None:

                #     shorten_sentences_indices = shorten_sentences_indices.to(extra_model.device)
                #     attention_mask = attention_mask.to(extra_model.device)

                #     if self.llm_output_layer == -1:
                #         out = extra_model(input_ids = shorten_sentences_indices, 
                #                         attention_mask = attention_mask,
                #                         output_attentions = False)
                #         description_encodings = out.last_hidden_state 
                #     else:
                #         out = extra_model(input_ids = shorten_sentences_indices, 
                #                         attention_mask = attention_mask,
                #                         output_attentions = False,
                #                         output_hidden_states = True)
                #     description_encodings = out.hidden_states[self.llm_output_layer]

                # else:

                inputs = self.llm_tokenizer(split_union_sentences, return_tensors="pt", padding = "longest")
                shorten_sentences_indices = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]

                shorten_sentences_indices = shorten_sentences_indices.to(self.llm_model.device)
                attention_mask = attention_mask.to(self.llm_model.device)
                

                if fold == 0:

                    description_encodings = self.llama_encoder(shorten_sentences_indices, attention_mask)

                else:

                    input_length = len(split_union_sentences) // fold
                    out = []
                    for fold_i in range(fold):

                        if fold_i != fold - 1:

                            
                            input_ids = shorten_sentences_indices[fold_i * input_length : (fold_i + 1) * input_length]
                            input_mask = attention_mask[fold_i * input_length : (fold_i + 1) * input_length]

                        else:

                            input_ids = shorten_sentences_indices[fold_i * input_length:]
                            input_mask = attention_mask[fold_i * input_length:]


                        out.append(self.llama_encoder(input_ids, input_mask))

                    
                    description_encodings = torch.cat(out)

            elif "longclip" in self.llm_name:

                input_ids = longclip.tokenize(split_union_sentences).to(self.device)
                attention_mask = (input_ids != 0)

                with torch.no_grad():
                    description_encodings = self.llm_model.encode_text(input_ids, early_stop = True)
            
            elif "clip" in self.llm_name:
 
                max_length = self.clip_length
                input_ids = self.llm_tokenize(split_union_sentences, context_length = max_length).to(self.device)
                attention_mask = (input_ids != 0)

                description_encodings = self.clip_encoder(input_ids, length=max_length)
            
            if self.fixed_embedding:
                description_encodings = description_encodings.detach().to(self.device)

            description_encodings = description_encodings.to(torch.float32)
            if self.intra_mlp is not None:
                N, L, _ = description_encodings.shape
                description_encodings = description_encodings.view(N * L, -1)
                description_encodings = self.intra_mlp(description_encodings)
                description_encodings = description_encodings.view(N, L, -1)

            description_encodings = description_encodings.permute(1,0,2)

            if self.mask:
                seq_length = description_encodings.shape[0]
                mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
                mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
                mask = mask.to(description_encodings.device)

            else:
                mask = None

            if self.version == "old":
                if self.intra_module is not None:
                    for idx in range(len(self.intra_module)):
                        description_encodings = self.intra_module[idx](description_encodings, mask, src_key_padding_mask = ~attention_mask.bool())
            else:
                description_encodings = self.intra_module(description_encodings, mask, src_key_padding_mask = ~attention_mask.bool())
            
            logits_dict = {}
            if self.intra_decoder is not None:
                    
                query_embed = self.query_embed.weight
                query_embed = query_embed.unsqueeze(1).repeat(1, description_encodings.shape[1], 1)
                tgt = torch.zeros_like(query_embed)

                if self.version == "old":
                    for idx in range(len(self.intra_decoder)):
                        tgt = self.intra_decoder[idx](tgt + query_embed, description_encodings, memory_key_padding_mask = ~attention_mask.bool())
                    description_encodings = tgt

                else:
                    description_encodings = self.intra_decoder(tgt + query_embed, description_encodings, memory_key_padding_mask = ~attention_mask.bool())
                
                if self.intra_loss:
                    logits_dict["pred_class"] = self.class_embed(description_encodings)
                    logits_dict["pred_color"] = self.color_embed(description_encodings)
                    logits_dict["pred_direction"] = self.direction_embed(description_encodings)
                

            description_encodings = description_encodings.permute(1,0,2).contiguous()

            if self.intra_proj == "max":
                description_encodings = description_encodings.max(dim = 1)[0]
            elif self.intra_proj == "last":
                description_encodings = description_encodings[:,-1]
            elif self.intra_proj == "mean":
                description_encodings = description_encodings.mean(dim = 1)


            if self.inter_mlp is not None:
                description_encodings = self.inter_mlp(description_encodings)
            
            if (length == [] or hints_number == []) and new_length == []:
                description_encodings = description_encodings.view(batch_size, num_sentence, -1)
                padding_mask = None
            else:
                new_description_encodings = torch.zeros(batch_size, num_sentence, description_encodings.shape[-1]).to(self.device)
                padding_mask = torch.ones(batch_size, num_sentence, dtype=attention_mask.dtype).to(self.device)
                start = 0
                end = 0
                if new_length != []:
                    for idx, l in enumerate(new_length):
                        end = start + l
                        new_description_encodings[idx, :l] = description_encodings[start:end]
                        padding_mask[idx, :l] = 0
                        start = end

                else:
                    for idx, l in enumerate(length):
                        end = start + l
                        new_description_encodings[idx, :l] = description_encodings[start:end]
                        padding_mask[idx, :l] = 0
                        start = end
                description_encodings = new_description_encodings

            if self.inter_pos_embedding != None:
                description_encodings = self.inter_pos_embedding(description_encodings)
            
            if self.is_fine:
                return description_encodings, logits_dict, padding_mask
            
            if inst_list != [] and inst_list[0] != None:
                N, L, C = description_encodings.shape
                L_I = len(inst_list[0])
                inst_encodings = torch.zeros(N, L_I, C).to(self.device)
                for i_sample, sample in enumerate(inst_list):
                    for i_inst, instance_idx in enumerate(sample):
                        inst_encodings[i_sample, i_inst] = description_encodings[i_sample, instance_idx]
                inst_encodings = inst_encodings.view(N * L_I, C)

            else:
                inst_encodings = None
            
            repetition = False
            if repetition:
                # assert hints_number != []
                # new_description_encodings = torch.zeros(batch_size, 8, description_encodings.shape[-1]).to(self.device)
                # for idx, l in enumerate(length):
                #     num_sum = 0
                #     for hint_idx, i in enumerate(hints_number[idx]):
                #         for _ in range(i):
                #             new_description_encodings[idx, num_sum] = description_encodings[idx, hint_idx]
                #             num_sum += 1
                # new_description_encodings = new_description_encodings.permute(1,0,2)
                # for idx in range(len(self.inter_module)):
                #     new_description_encodings += self.inter_module[idx](new_description_encodings)
                # description_encodings = new_description_encodings
                pass

            else:
                description_encodings = description_encodings.permute(1,0,2)
                if self.version == "old":
                    if self.inter_module is not None:
                        for idx in range(len(self.inter_module)):
                            # description_encodings = description_encodings.clone() + \
                            #     self.inter_module[idx](description_encodings, src_key_padding_mask = padding_mask.bool() if padding_mask is not None else None)
                            description_encodings = self.inter_module[idx](description_encodings, src_key_padding_mask = padding_mask.bool() if padding_mask is not None else None)
                else:
                    self.inter_module(description_encodings, src_key_padding_mask = padding_mask.bool() if padding_mask is not None else None)
            
            if (length == [] or hints_number == []) or repetition:
                if self.inter_proj == "max":
                    description_encodings = description_encodings.max(dim = 0)[0]
                elif self.inter_proj == "mean":
                    description_encodings = description_encodings.mean(dim = 0)
            else:
                new_description_encodings = torch.zeros(batch_size, description_encodings.shape[-1]).to(self.device)
                description_encodings = description_encodings.permute(1,0,2)

                for idx, l in enumerate(length):
                    if self.inter_proj == "max":
                        new_description_encodings[idx] = description_encodings[idx, :l].max(dim = 0)[0]
                    elif self.inter_proj == "mean":
                        new_description_encodings[idx] = description_encodings[idx, :l].mean(dim = 0)
                description_encodings = new_description_encodings

            # for para in self.llm_model.parameters():
            #     print(para)
            # description_encodings = F.normalize(description_encodings, dim = -1)

            return description_encodings, logits_dict, inst_encodings

        elif self.split_type == "all":
            
            batch_size = len(descriptions)
            # num_sentence = len(split_union_sentences) // batch_size if length == [] else max(length)

            if "t5" in self.llm_name:
                inputs = self.llm_tokenizer(descriptions, return_tensors="pt", padding = "longest")
                shorten_sentences_indices = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]

                shorten_sentences_indices = shorten_sentences_indices.to(self.device)
                attention_mask = attention_mask.to(self.device)
                with torch.no_grad():
                    out = self.llm_model(input_ids = shorten_sentences_indices, 
                                    attention_mask = attention_mask,
                                    output_attentions = False)
                    description_encodings = out.last_hidden_state
            
            elif "llama" in self.llm_name:
                inputs = self.llm_tokenizer(descriptions, return_tensors="pt", padding = "longest")
                shorten_sentences_indices = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]

                shorten_sentences_indices = shorten_sentences_indices.to(self.device)
                attention_mask = attention_mask.to(self.device)
                if self.llm_output_layer == -1:
                    out = self.llm_model(input_ids = shorten_sentences_indices, 
                                    attention_mask = attention_mask,
                                    output_attentions = False)
                    description_encodings = out.last_hidden_state 
                else:
                    out = self.llm_model(input_ids = shorten_sentences_indices, 
                                    attention_mask = attention_mask,
                                    output_attentions = False,
                                    output_hidden_states = True)
                    description_encodings = out.hidden_states[self.llm_output_layer]

            if self.fixed_embedding:
                description_encodings = description_encodings.detach()

            if self.intra_mlp is not None:
                N, L, _ = description_encodings.shape
                description_encodings = description_encodings.view(N * L, -1)
                description_encodings = self.intra_mlp(description_encodings)
                description_encodings = description_encodings.view(N, L, -1)

            description_encodings = description_encodings.permute(1,0,2)

            if self.mask:
                seq_length = description_encodings.shape[0]
                mask = torch.triu(torch.ones(seq_length, seq_length), diagonal=1)
                mask = mask.masked_fill(mask == 1, float('-inf')).masked_fill(mask == 0, float(0.0))
                mask = mask.to(description_encodings.device)

            else:
                mask = None

            if self.version == "old":
                for idx in range(len(self.intra_module)):
                    description_encodings = self.intra_module[idx](description_encodings, mask, src_key_padding_mask = ~attention_mask.bool())
            else:
                description_encodings = self.intra_module(description_encodings, mask, src_key_padding_mask = ~attention_mask.bool())
            
            logits_dict = {}
            if self.intra_decoder is not None:
                    
                query_embed = self.query_embed.weight
                query_embed = query_embed.unsqueeze(1).repeat(1, description_encodings.shape[1], 1)
                tgt = torch.zeros_like(query_embed)

                if self.version == "old":
                    for idx in range(len(self.intra_decoder)):
                        tgt = self.intra_decoder[idx](tgt + query_embed, description_encodings, memory_key_padding_mask = ~attention_mask.bool())
                    description_encodings = tgt

                else:
                    description_encodings = self.intra_decoder(tgt + query_embed, description_encodings, memory_key_padding_mask = ~attention_mask.bool())
                
                if self.intra_loss:
                    logits_dict["pred_class"] = self.class_embed(description_encodings).permute(1,0,2)
                    logits_dict["pred_color"] = self.color_embed(description_encodings).permute(1,0,2)
                    logits_dict["pred_direction"] = self.direction_embed(description_encodings).permute(1,0,2)             

            description_encodings = description_encodings.permute(1,0,2).contiguous()

            if self.intra_proj == "max":
                description_encodings = description_encodings.max(dim = 1)[0]
            elif self.intra_proj == "mean":
                description_encodings = description_encodings.mean(dim = 1)
            elif self.intra_proj == "last":
                description_encodings = description_encodings[:,-1]


            if self.inter_mlp is not None:
                description_encodings = self.inter_mlp(description_encodings)

            return description_encodings, logits_dict, None 


        else:
            raise NameError(f"The split type {self.split_type} is not valid.")

    @property
    def device(self):
        if self.inter_mlp is not None:
            return next(self.inter_mlp.parameters()).device
        else:
            return next(self.intra_mlp.parameters()).device