from typing import List
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import transforms, models

import numpy as np

from models.language_encoder import get_mlp
from models.pointcloud.pointnet2 import PointNet2

from datapreparation.kitti360pose.imports import Object3d
from configs.mixed0 import COLOR_NAMES

import matplotlib.pyplot as plt

def save_convert_image(tensor, pic_address = "image_from_tensor.png"):
    
    image = tensor.detach().cpu().permute(1, 2, 0)  # Change to H x W x C for matplotlib

    # Plot and save the image
    plt.imshow(image.numpy())  # Convert to NumPy array
    plt.axis('off')  # Remove axis
    plt.savefig(pic_address, bbox_inches='tight', pad_inches=0)  # Save as a PNG file
    plt.close()


class ObjectEncoder(torch.nn.Module):
    def __init__(self, embed_dim: int, known_classes: List[str], known_colors: List[str], args, is_eval = False):
        """Module to encode a set of objects

        Args:
            embed_dim (int): Embedding dimension
            known_classes (List[str]): List of known classes, only used for embedding ablation
            known_colors (List[str]): List of known colors, only used for embedding ablation
            args: Global training arguments
        """
        super(ObjectEncoder, self).__init__()

        self.embed_dim = embed_dim
        self.args = args

        # Set idx=0 for padding
        self.known_classes = {c: (i + 1) for i, c in enumerate(known_classes)}
        self.known_classes["<unk>"] = 0
        self.class_embedding = nn.Embedding(len(self.known_classes), embed_dim, padding_idx=0)
        
        self.known_colors = {c: i for i, c in enumerate(COLOR_NAMES)}
        self.known_colors["<unk>"] = 0
        self.color_embedding = nn.Embedding(len(self.known_colors), embed_dim, padding_idx=0)

        self.pos_encoder = get_mlp([3, 64, embed_dim])  # OPTION: pos_encoder layers
        self.color_encoder = get_mlp([3, 64, embed_dim])  # OPTION: color_encoder layers
        self.num_encoder = get_mlp([1, 64, embed_dim])

        self.num_mean = 1826.6844940968194
        self.num_std = 2516.8905096993817

        self.pointnet = PointNet2(
            len(known_classes), len(known_colors), args
        )  # The known classes are all the same now, at least for K360
        if not is_eval and args.pointnet_path != "":
            pointnet_dict = torch.load(args.pointnet_path)

            key_list = []

            for key in pointnet_dict:
                if "classifier" in key:
                    value = pointnet_dict[key]
                    fea_list = key.split(".")
                    assert len(fea_list) == 2
                    fea1, fea2 = fea_list[0], fea_list[1]
                    if getattr(getattr(self.pointnet, fea1), fea2).shape != value.shape:
                        key_list.append(key)
                        print(getattr(getattr(self.pointnet, fea1), fea2).shape)
                        print(value.shape)
                            
            conflict_list = []
            if key_list != []:
                for key in key_list:             
                    pointnet_dict.move_to_end(key)
                    item = pointnet_dict.popitem()
                    conflict_list.append({item[0]: item[1]})
                
                self.pointnet.load_state_dict(pointnet_dict, strict = False)

                # for items in conflict_list:
                #     for key, value in items.items():
                #         fea_list = key.split(".")
                #         fea1, fea2 = fea_list[0], fea_list[1]
                #         num_classes = getattr(getattr(self.pointnet, fea1), fea2).shape[0]
                #         num_dim = value.dim
                #         assert num_classes > value.shape[0]
                #         assert num_dim in [1, 2]

                #         if num_dim == 1:
                #             value2 = torch.zeros(num_classes - value.shape[0])
                #         else:
                #             value2 = torch.zeros(num_classes - value.shape[0], value.shape[1])

                #         init_value = getattr(getattr(self.pointnet, fea1), fea2)
                #         value2 = init_value[num_classes:]
                #         value = torch.cat([value, value2])
                #         getattr(getattr(self.pointnet, fea1), fea2) = value


            else:
                self.pointnet.load_state_dict(pointnet_dict)
        
        if args.is_global:
            self.global_pointnet = PointNet2(
                len(known_classes), len(known_colors), args, is_global=True
            )
            if args.pointnet_features == 0:
                self.mlp_global_pointnet = get_mlp([self.global_pointnet.dim0, self.embed_dim])
            elif args.pointnet_features == 1:
                self.mlp_global_pointnet  = get_mlp([self.global_pointnet.dim1, self.embed_dim])
            elif args.pointnet_features == 2:
                self.mlp_global_pointnet  = get_mlp([self.global_pointnet.dim2, self.embed_dim])
        # self.pointnet_dim = self.pointnet.lin2.weight.size(0)

        if args.pointnet_freeze:
            print("CARE: freezing PN")
            self.pointnet.requires_grad_(False)
        

        if args.pointnet_features == 0:
            self.mlp_pointnet = get_mlp([self.pointnet.dim0, self.embed_dim])
        elif args.pointnet_features == 1:
            self.mlp_pointnet = get_mlp([self.pointnet.dim1, self.embed_dim])
        elif args.pointnet_features == 2:
            self.mlp_pointnet = get_mlp([self.pointnet.dim2, self.embed_dim])

        self.mlp_merge = get_mlp([len(args.use_features) * embed_dim, embed_dim])

        if "image" in args.use_features:
            self.transform = transforms.Compose([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
            if args.image_version == "resnet18":
                self.image_model = models.resnet18(pretrained=True)
            elif args.image_version == "resnet34":
                self.image_model = models.resnet34(pretrained=True)
            else:
                raise NameError("No such resnet version")
            cnn_output_dim = self.image_model.fc.weight.shape[1]
            self.image_model.fc = nn.Sequential()
            self.image_size = 224
            self.image_mlp = get_mlp([cnn_output_dim, self.embed_dim])
            self.fix_image = args.fix_image
            self.image_version = args.image_version

            if args.fix_image:
                for para in self.image_model.parameters():
                    para.requires_grad = False

        else:
            self.image_model = None

    def forward(self, objects: List[Object3d], object_points, global_points = None, pos_location=None):
        """Features are currently normed before merging but not at the end.

        Args:
            objects (List[List[Object3d]]): List of lists of objects
            object_points (List[Batch]): List of PyG-Batches of object-points
        """

        if ("class_embed" in self.args and self.args.class_embed) or (
            "color_embed" in self.args and self.args.color_embed
        ):
            class_indices = []
            color_indices = []
            for i_batch, objects_sample in enumerate(objects):
                for obj in objects_sample:
                    class_idx = self.known_classes.get(obj.label, 0)
                    class_indices.append(class_idx)
                    color_idx = self.known_colors[obj.get_color_text()]
                    color_indices.append(color_idx)

        if "class_embed" not in self.args or self.args.class_embed == False:
            # Void all colors for ablation
            # if "color" not in self.args.use_features:
            #     for pyg_batch in object_points:
            #         pyg_batch.x[:] = 0.0  # x is color, pos is xyz

            out = [
                self.pointnet(pyg_batch.to(self.get_device()))
                for pyg_batch in object_points
            ]  # [B, obj_counts, PN_dim]
            object_features = [out_d.features2 for out_d in out]
            class_pred = [out_d.class_pred for out_d in out]

            object_features = torch.cat(object_features, dim=0)  # [total_objects, PN_dim]
            object_features = self.mlp_pointnet(object_features)
        else:
            class_pred = None

        embeddings = []
        if "class" in self.args.use_features:
            if (
                "class_embed" in self.args and self.args.class_embed
            ):  # Use fixed embedding (ground-truth data!)
                class_embedding = self.class_embedding(
                    torch.tensor(class_indices, dtype=torch.long, device=self.get_device())
                )
                embeddings.append(F.normalize(class_embedding, dim=-1))
            else:
                embeddings.append(
                    F.normalize(object_features, dim=-1)
                )  # Use features from PointNet

        if "color" in self.args.use_features:
            if "color_embed" in self.args and self.args.color_embed:
                color_embedding = self.color_embedding(
                    torch.tensor(color_indices, dtype=torch.long, device=self.get_device())
                )
                embeddings.append(F.normalize(color_embedding, dim=-1))
            else:
                colors = []
                for objects_sample in objects:
                    colors.extend([obj.get_color_rgb() for obj in objects_sample])
                color_embedding = self.color_encoder(
                    torch.tensor(colors, dtype=torch.float, device=self.get_device())
                )
                embeddings.append(F.normalize(color_embedding, dim=-1))

        if "position" in self.args.use_features:
            positions = []
            for objects_sample in objects:
                positions.extend([obj.get_center() for obj in objects_sample])
            pos_positions = torch.tensor(positions, dtype=torch.float, device=self.get_device())
            pos_embedding = self.pos_encoder(pos_positions)
            embeddings.append(F.normalize(pos_embedding, dim=-1))
        else:
            pos_positions = None

        if "num" in self.args.use_features:
            num_points = []
            for objects_sample in objects:
                if hasattr(objects_sample[0], "num"):
                    num_points.extend([obj.num for obj in objects_sample])
                else:
                    num_points.extend([len(obj.xyz) for obj in objects_sample])
            num_points_embedding = self.num_encoder(
                (torch.tensor(num_points, dtype=torch.float, device=self.get_device()).unsqueeze(-1) - self.num_mean) / self.num_std
            )
            embeddings.append(F.normalize(num_points_embedding, dim=-1))

        if self.image_model != None:
            input_objects = []
            for obj in objects:
                for instance in obj:
                    pc = torch.clamp(torch.FloatTensor(instance.xyz)[:,:2], 0 ,1 - 1e-7)
                    pc = pc.to(self.device)
                    image = torch.zeros(self.image_size, self.image_size).to(self.device)
                    
                    pc = torch.floor(pc * self.image_size ).long()
                    image[pc[:,0], pc[:,1]] = 1
                    image = image.unsqueeze(0).repeat(3, 1, 1)
                    inputs = self.transform(image)
                    input_objects.append(inputs)

            input_objects = torch.stack(input_objects)
            if self.image_version == "resnet34":
                if self.fix_image:

                    with torch.no_grad():
                        output_objects = self.image_model(input_objects)

                else:
                    len_input_objects = len(input_objects)
                    # if len_input_objects < 1000:
                    #     image_fold = 10
                    # elif len_input_objects < 2000:
                    #     image_fold = 20
                    # else:
                    #     image_fold = 30
                    image_length = 24
                    image_fold = len_input_objects // image_length + (len_input_objects % image_length != 0)
                    output_part_objects = []

                    for image_i in range(image_fold):
                        if image_i != image_fold - 1:
                            input_part_objects = input_objects[(image_i * image_length) : (image_i * image_length + image_length)]
                        else:
                            print(len_input_objects - image_i * image_length)
                            input_part_objects = input_objects[(image_i * image_length):]
                        output_part_objects.append(self.image_model(input_part_objects))
                    output_objects = torch.cat(output_part_objects)


            
            else:
                if self.fix_image:
                    with torch.no_grad():
                        output_objects = self.image_model(input_objects)
                else:
                    output_objects = self.image_model(input_objects)
            output_objects = self.image_mlp(output_objects)
            embeddings.append(output_objects)


        if len(embeddings) > 1:
            embeddings = self.mlp_merge(torch.cat(embeddings, dim=-1))
        else:
            embeddings = embeddings[0]

        if hasattr(self, "global_pointnet"):
            object_features = [
                self.global_pointnet(pyg_batch.to(self.get_device())).features2
                for pyg_batch in global_points
            ]  # [B, PN_dim]

            object_features = torch.cat(object_features, dim=0)  # [total_objects, PN_dim]
            global_embeddings = self.mlp_global_pointnet(object_features)
        else:
            global_embeddings = None

        return embeddings, pos_positions, global_embeddings, class_pred

    @property
    def device(self):
        return next(self.class_embedding.parameters()).device

    def get_device(self):
        return next(self.class_embedding.parameters()).device
