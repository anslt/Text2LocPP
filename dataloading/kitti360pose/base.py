from typing import List

import os
import os.path as osp
import pickle
import numpy as np
import cv2
import json

import torch
from torch.utils.data import Dataset, DataLoader

from configs.mixed0  import (
    CLASS_TO_LABEL,
    LABEL_TO_CLASS,
    CLASS_TO_MINPOINTS,
    SCENE_NAMES,
    CLASS_TO_INDEX,
    COLOR_TO_INDEX,
    DIRECTION_TO_INDEX,
)

from configs.mixed0 import SCENE_NAMES, SCENE_NAMES_TRAIN, SCENE_NAMES_VAL, SCENE_NAMES_TEST
from datapreparation.kitti360pose.imports import Object3d, Cell, Pose
from datapreparation.kitti360pose.drawing import (
    show_pptk,
    show_objects,
    plot_cell,
    plot_pose_in_best_cell,
)

def opposite_direction(d):
    if d == "north":
        return "south"
    elif d == "south":
        return "north"
    elif d == "east":
        return "west"
    elif d == "west":
        return "east"
    return d

def article(d):
    if d == "east":
        return "an"
    else:
        return "a"

def article_big(d):
    if d == "east":
        return "An"
    else:
        return "A"

class Kitti360BaseDataset(Dataset):
    def __init__(self, base_path, scene_name, add_object_number=False, no_color = False, text_mode = 0, opposite = False):
        """Base dataset for loading Kitti360Pose data.

        Args:
            base_path: Base path for the Kitti360Pose scenes
            scene_name: Name of the scene to load
        """
        self.scene_name = scene_name
        self.no_color = no_color

        if not osp.exists(osp.join(base_path, "cells", f"{scene_name}.pkl")):
            base_path = "./data/k360_30-10_scG_pd10_pc4_spY_all"
            assert osp.exists(osp.join(base_path, "cells", f"{scene_name}.pkl"))
        self.cells = pickle.load(
            open(osp.join(base_path, "cells", f"{scene_name}.pkl"), "rb")
        )  # Also use objects from here for classification
        self.cells_dict = {cell.id: cell for cell in self.cells}
        cell_ids = [cell.id for cell in self.cells]
        assert len(np.unique(cell_ids)) == len(cell_ids)

        if osp.exists(osp.join(base_path, "poses1", f"{scene_name}.pkl")):
            self.poses = pickle.load(open(osp.join(base_path, "poses1", f"{scene_name}.pkl"), "rb"))
            print("POSE 1")
        else:
            self.poses = pickle.load(open(osp.join(base_path, "poses", f"{scene_name}.pkl"), "rb"))
            print("POSE 0")
            
        if no_color and osp.samefile(base_path, "./data/k360_30-10_scG_pd10_pc4_spY_all"):
            if osp.exists(osp.join(base_path, "texts2", f"{scene_name}.json")) and text_mode == 1:
                assert osp.exists(osp.join(base_path, "texts1", f"{scene_name}_label.json"))
                with open(osp.join(base_path, "texts2", f"{scene_name}.json"), "r") as d:
                    self.med_texts = json.load(d)
                    d.close()
                with open(osp.join(base_path, "texts1", f"{scene_name}_label.json"), "r") as d:
                    self.med_texts_label = json.load(d)
                    d.close()
            else:
                self.med_texts = []

        else:
            if osp.exists(osp.join(base_path, "texts1", f"{scene_name}.json")) and text_mode == 1:
                assert osp.exists(osp.join(base_path, "texts1", f"{scene_name}_label.json"))
                with open(osp.join(base_path, "texts1", f"{scene_name}.json"), "r") as d:
                    self.med_texts = json.load(d)
                    d.close()
                with open(osp.join(base_path, "texts1", f"{scene_name}_label.json"), "r") as d:
                    self.med_texts_label = json.load(d)
                    d.close()
            else:
                self.med_texts = []

        if no_color and osp.samefile(base_path, "./data/k360_30-10_scG_pd10_pc4_spY_all"):
            if osp.exists(osp.join(base_path, "gpt_texts2", f"{scene_name}.json")) and text_mode == 2:
                with open(osp.join(base_path, "gpt_texts2", f"{scene_name}.json"), "r") as d:
                    self.hard_texts = json.load(d)
                    d.close()
            elif osp.exists(osp.join(base_path, "gpt_texts2", "text0_new", f"{scene_name}.json")) and text_mode == 4:
                with open(osp.join(base_path, "gpt_texts2", "text0_new", f"{scene_name}.json"), "r") as d:
                    self.hard_texts = json.load(d)
                    d.close()
            else:
                self.hard_texts = []
        else:
            if osp.exists(osp.join(base_path, "gpt_texts1", f"{scene_name}.json")) and text_mode == 2:
                with open(osp.join(base_path, "gpt_texts1", f"{scene_name}.json"), "r") as d:
                    self.hard_texts = json.load(d)
                    d.close()
            elif osp.exists(osp.join(base_path, "gpt_texts1", "text0", f"{scene_name}.json")) and text_mode == 3:
                with open(osp.join(base_path, "gpt_texts1", "text0", f"{scene_name}.json"), "r") as d:
                    self.hard_texts = json.load(d)
                    d.close()
            elif osp.exists(osp.join(base_path, "gpt_texts1", "text0_t5-small", f"{scene_name}.json")) and text_mode == 4:
                with open(osp.join(base_path, "gpt_texts1", "text0_t5-small", f"{scene_name}.json"), "r") as d:
                    self.hard_texts = json.load(d)
                    d.close()
            else:
                self.hard_texts = []
    
        # if base_path != "./data/k360_30-10_scG_pd10_pc4_spY_all":
        #     import pdb; pdb.set_trace()
        if no_color and osp.samefile(base_path, "./data/k360_30-10_scG_pd10_pc4_spY_all"):
            if osp.exists(osp.join(base_path, "texts5", f"{scene_name}.json")) and text_mode == 2:
                with open(osp.join(base_path, "texts5", f"{scene_name}.json"), "r") as d:
                    self.hard_texts_help = json.load(d)
                    d.close()
            else:
                self.hard_texts_help = []
        else:
            if osp.exists(osp.join(base_path, "texts4", f"{scene_name}.json")) and text_mode == 2:
                with open(osp.join(base_path, "texts4", f"{scene_name}.json"), "r") as d:
                    self.hard_texts_help = json.load(d)
                    d.close()
            else:
                self.hard_texts_help = []
        
        # print("cell_size:", len(self.cells))
        # print("pose_size:", len(self.poses))
  
        self.class_to_index = CLASS_TO_INDEX
        self.hint_descriptions, self.hint_index = zip(*[
            Kitti360BaseDataset.create_hint_description(pose, self.cells_dict[pose.cell_id], no_color, opposite=opposite)
            for pose in self.poses
        ])

        if hasattr(self.poses[0], "information"):
            self.hint1_descriptions, self.hint1_number, self.hint1_index = zip(*[
                Kitti360BaseDataset.create_hint1_description(pose, add_object_number, no_color=no_color)
                for pose in self.poses
            ])
        else:
            self.hint1_descriptions = []
            self.hint1_number = []


    def __getitem__(self, idx):
        raise Exception("Not implemented: abstract class.")

    def create_hint_description(pose: Pose, cell: Cell, no_color = False, opposite = False):
        hints = []
        index = []
        for descr in pose.descriptions:
            # obj = cell_objects_dict[descr.object_id]
            # hints.append(f'The pose is {descr.direction} of a {obj.get_color_text()} {obj.label}.')

            if opposite:

                if no_color:

                    hints.append(
                        # f"The pose is {descr.direction} of a {descr.object_color_text} {descr.object_label}."
                            f"A {descr.object_label} is {opposite_direction(descr.direction)} of the pose."
                    )
                    index.append((DIRECTION_TO_INDEX[opposite_direction(descr.direction)], 
                                CLASS_TO_INDEX[descr.object_label]))


                else:           
                    hints.append(
                        # f"The pose is {descr.direction} of a {descr.object_color_text} {descr.object_label}."
                            f"A {descr.object_color_text} {descr.object_label} is {opposite_direction(descr.direction)} of the pose."
                    )
                    index.append((DIRECTION_TO_INDEX[opposite_direction(descr.direction)], 
                                COLOR_TO_INDEX[descr.object_color_text],
                                CLASS_TO_INDEX[descr.object_label]))
                    
            else:
             
                if no_color:

                    hints.append(
                        # f"The pose is {descr.direction} of a {descr.object_color_text} {descr.object_label}."
                            f"The pose is {descr.direction} of a {descr.object_label}."
                    )
                    index.append((DIRECTION_TO_INDEX[descr.direction], 
                                CLASS_TO_INDEX[descr.object_label]))


                else:           
                    hints.append(
                        # f"The pose is {descr.direction} of a {descr.object_color_text} {descr.object_label}."
                            f"The pose is {descr.direction} of a {descr.object_color_text} {descr.object_label}."
                    )
                    index.append((DIRECTION_TO_INDEX[descr.direction], 
                                COLOR_TO_INDEX[descr.object_color_text],
                                CLASS_TO_INDEX[descr.object_label]))

        return hints, index
        
    def create_hint1_description(pose: Pose, add_object_number: bool, shuffle: bool = False, no_color = False):
        hints = []
        hints_number = []
        index = []
        for key, value in pose.information.items():
            descrs = pose.descriptions
            if key in ["near", "far"]:
                if value == -1:
                    continue
                if no_color:
                    color_info = ""
                else:
                    color_info = descrs[value].object_color_text
                direction = descrs[value].direction
                direction = opposite_direction(direction)
                suffix = "est" if key == "near" else "thest"
                text = f"The {key}{suffix} instance is {article(direction)} {direction}ern {color_info} {descrs[value].object_label}."
                hints.append(text)
                hint_number = 1
            else:
                if value != []:
                    text = ""
                    text0 = []
                    length = len(value)
                    hint_number = length
                    if length > 1 and shuffle:
                        value1 = np.random.choice(value, size=length, replace=False)
                    else:
                        value1 = value
                    for ele in value1:
                        if no_color:
                            color_info = ""
                        else:
                            color_info = descrs[ele].object_color_text + " "
                        if text0 == [] and not add_object_number:
                            text0.append(f"A {color_info}{descrs[ele].object_label}")
                        else:
                            text0.append(f"a {color_info}{descrs[ele].object_label}")

                        if no_color:
                            index.append((DIRECTION_TO_INDEX[opposite_direction(key)], 
                                        CLASS_TO_INDEX[descrs[ele].object_label]))

                        else:

                            index.append((DIRECTION_TO_INDEX[opposite_direction(key)],
                                        COLOR_TO_INDEX[descrs[ele].object_color_text], 
                                        CLASS_TO_INDEX[descrs[ele].object_label]))
                        
                    if add_object_number:
                        if length == 1:
                            text = f"There is 1 object, {text0[0]}, is {opposite_direction(key)} of the pose."
                        elif length == 2:
                            text = f"There are 2 objects, {text0[0]} and {text0[1]}, are {opposite_direction(key)} of the pose."
                        else:
                            text_prefix = f"There are {str(length)} objects, "
                            text_suffix = f", are {opposite_direction(key)} of the pose."
                            for idx in range(length):
                                if idx == 0:
                                    text_in = text0[idx]
                                elif idx == length - 1:
                                    text_in += ", and " + text0[idx]
                                else:
                                    text_in += ", " + text0[idx]
                            text = text_prefix + text_in + text_suffix
                    else:
                        if length == 1:
                            text = text0[0] + f" is {opposite_direction(key)} of the pose."
                        elif length == 2:
                            text = text0[0] + " and " + text0[1] + f" is {opposite_direction(key)} of the pose."
                        else:
                            for idx in range(length):
                                if idx == 0:
                                    text += text0[idx]
                                elif idx == length - 1:
                                    text += ", and " + text0[idx] + f" is {opposite_direction(key)} of the pose."
                                else:
                                    text += ", " + text0[idx]
                    hints.append(text)
                else:
                    continue
            hints_number.append(hint_number)
        return hints, hints_number, index


    def get_known_classes(self):
        return list(self.class_to_index.keys())

    def get_known_words(self):
        words = []
        for hints in self.hint_descriptions:
            for hint in hints:
                words.extend(hint.replace(".", "").replace(",", "").lower().split())
        return list(np.unique(words))

    def __len__(self):
        raise Exception("Not implemented: abstract class.")

    def collate_fn(data):
        batch = {}
        for key in data[0].keys():
            batch[key] = [data[i][key] for i in range(len(data))]
        return batch


class Kitti360BaseDataset_old(Dataset):
    def __init__(self, base_path, scene_name):
        """Base dataset for loading Kitti360Pose data.

        Args:
            base_path: Base path for the Kitti360Pose scenes
            scene_name: Name of the scene to load
        """
        self.scene_name = scene_name
        self.cells = pickle.load(
            open(osp.join(base_path, "cells", f"{scene_name}.pkl"), "rb")
        )  # Also use objects from here for classification
        self.cells_dict = {cell.id: cell for cell in self.cells}

        cell_ids = [cell.id for cell in self.cells]
        assert len(np.unique(cell_ids)) == len(cell_ids)

        self.poses = pickle.load(open(osp.join(base_path, "poses", f"{scene_name}.pkl"), "rb"))

        self.class_to_index = CLASS_TO_INDEX

        self.hint_descriptions = [
            Kitti360BaseDataset.create_hint_description(pose, self.cells_dict[pose.cell_id])
            for pose in self.poses
        ]

    def __getitem__(self, idx):
        raise Exception("Not implemented: abstract class.")

    def create_hint_description(pose: Pose, cell: Cell):
        hints = []
        # cell_objects_dict = {obj.id: obj for obj in cell.objects}
        for descr in pose.descriptions:
            # obj = cell_objects_dict[descr.object_id]
            # hints.append(f'The pose is {descr.direction} of a {obj.get_color_text()} {obj.label}.')
            hints.append(
                f"The pose is {descr.direction} of a {descr.object_color_text} {descr.object_label}."
            )
        return hints

    def get_known_classes(self):
        return list(self.class_to_index.keys())

    def get_known_words(self):
        words = []
        for hints in self.hint_descriptions:
            for hint in hints:
                words.extend(hint.replace(".", "").replace(",", "").lower().split())
        return list(np.unique(words))

    def __len__(self):
        raise Exception("Not implemented: abstract class.")

    def collate_fn(data):
        batch = {}
        for key in data[0].keys():
            batch[key] = [data[i][key] for i in range(len(data))]
        return batch

if __name__ == "__main__":
    base_path = "./data/k360_decouple"
    folder_name = "2013_05_28_drive_0003_sync"

    dataset = Kitti360BaseDataset(base_path, folder_name)
