from typing import List

import os
import os.path as osp
import pickle
import numpy as np
import cv2
from copy import deepcopy

import torch
from torch.utils.data import Dataset, DataLoader

import torch_geometric.transforms as T

from configs.mixed0 import (
    CLASS_TO_LABEL,
    LABEL_TO_CLASS,
    CLASS_TO_INDEX,
    CLASS_TO_MINPOINTS,
    SCENE_NAMES,
    SCENE_NAMES_TEST,
    SCENE_NAMES_TRAIN,
    SCENE_NAMES_VAL,
    COLOR_TO_INDEX,
    COLORS,
    DIRECTION_TO_INDEX,
)
from configs.mixed0 import CLASS_TO_INDEX, COLOR_NAMES
from datapreparation.kitti360pose.imports import Object3d, Cell, Pose
from datapreparation.kitti360pose.drawing import (
    show_pptk,
    show_objects,
    plot_cell,
    plot_pose_in_best_cell,
)
from datapreparation.kitti360pose.select import (
    get_direction,
    get_direction_noOntop,
    select_objects_closest,
    select_objects_direction,
    select_objects_class,
    select_objects_random,
    select_objects_direction_random,
)
from datapreparation.kitti360pose.imports import (
    Object3d,
    Cell,
    Pose,
    DescriptionPoseCell,
    DescriptionBestCell,
)
from datapreparation.kitti360pose.update import add_pose_information

from dataloading.kitti360pose.base import Kitti360BaseDataset
from dataloading.kitti360pose.utils import batch_object_points, flip_pose_in_cell, flip_pose_in_cell_easy, batch_global_points
from torch_geometric.data import Data, Batch
from nltk import tokenize as text_tokenize

try:
    from configs.kitti360 import KNOWN_CLASS as robust_class_list
    from configs.kitti360 import COLOR_NAMES as robust_color_list
    from configs.mixed0 import DIRECTION_TO_INDEX as robust_direction_dict
    robust_direction_list = [ele for ele in robust_direction_dict.keys()]
except:
    print("The robust test setting is fail.")

from data_gen.generate_moderate_text import generate_moderate_text
import tqdm

def poses_equivalent(p, r):
    return (
        p.cell_id == r.cell_id
        and np.allclose(p.pose_w, r.pose_w)
        and p.get_text() == r.get_text()
    )


def _description_same_object_id(d1, d2) -> bool:
    """Match by per-cell object_id when present; else fall back to instance_id."""
    o1 = getattr(d1, "object_id", None)
    o2 = getattr(d2, "object_id", None)
    if o1 is not None and o2 is not None:
        return o1 == o2
    return d1.object_instance_id == d2.object_instance_id


def merge_repl_descriptions_with_pose_by_object_id(pose: Pose, repl: Pose) -> Pose:
    """Align by index up to len(pose.descriptions). Same object_id -> keep pose's description; else repl's.
    If repl has more descriptions than pose, append the tail of repl unchanged.
    If repl is shorter, remaining pose slots are taken from pose."""
    pd = pose.descriptions
    rd = repl.descriptions
    merged = []
    unfound = []
    object_list = []
    try:
        for i in range(len(rd)):
            # import pdb; pdb.set_trace()
            if i < len(pd):
                if _description_same_object_id(pd[i], rd[i]):
                    merged.append(pd[i])
                else:
                    merged.append(pd[i])
                    unfound.append(rd[i])
                object_list.append(getattr(pd[i], "object_id", None))
            else:
                if unfound:
                    while unfound:
                        object_list.append(getattr(unfound[0], "object_id", None))
                        merged.append(unfound.pop(0))
                if getattr(rd[i], "object_id", None) or rd[i].object_id not in object_list:
                    merged.append(rd[i])
                    object_list.append(rd[i].object_id)
        if len(rd) < len(pd):
            print("rd is longer than pd")
            raise ValueError("rd is longer than pd")
        merged = merged[:len(rd)]
        # merged.extend(rd[len(pd) :])
    except:
        import pdb; pdb.set_trace()

    return Pose(
        repl.pose,
        repl.pose_w,
        repl.cell_id,
        repl.scene_name,
        merged,
        described_by=repl.described_by,
    )

CLASS_TO_PROB = {
    "building": 0.5,
    "pole": 0.6,
    "traffic light": 0.6,
    "traffic sign": 0.6,
    "garage": 0.5,
    "stop": 0.6,
    "smallpole": 0.6,
    "lamp": 0.6,
    "trash bin": 0.6,
    "vending machine": 0.6,
    "box": 0.6,
    "road": 0.1,
    "sidewalk": 0.3,
    "parking": 0.3,
    "wall": 0.5,
    "fence": 0.5,
    "guard rail": 0.5,
    "bridge":0.5,
    "tunnel": 0.5,
    "vegetation": 0.6,
    "terrain": 0.5,
    "dynamic": 0.6,
    "ground": 0.6,
    "static": 0.6,
    "water": 0.3,
    "stairs": 0.6,
    "pad": 1.0,
}

def describe_pose_in_pose_cell(
    pose_w: np.ndarray, cell: Cell, select_by, num_mentioned, max_dist=0.5, no_ontop=False
) -> List[DescriptionPoseCell]:
    """Describes a pose in a given cell.

    Args:
        pose_w (np.ndarray): Pose in world coordinates
        cell (Cell): Cell object
        select_by (str): How to select the objects to use for hints.
        num_mentioned (int): How many objects to mention.
        max_dist (float, optional): How far an object can be from the pose relative to the cell-size. Defaults to 0.5.
        no_ontop (bool, optional): Do not use the "on-top" direction. Defaults to False.

    Returns:
        List[DescriptionPoseCell]: List of DescriptionPoseCell structs. Can be used to create the actual text.
    """
    # Assert pose is close to cell_center
    # assert np.allclose(pose_w, cell.get_center())
    assert (
        len(cell.objects) >= num_mentioned
    ), f"Only {len(cell.objects)} objects, expected at least {num_mentioned}"

    # Norm pose
    pose = (pose_w - cell.bbox_w[0:3]) / cell.cell_size
    assert np.all(pose >= 0) and np.all(pose <= 1.0), f"{pose} {pose_w} {cell.bbox_w}"

    # Select candidates based on the distance
    dists = np.linalg.norm([obj.get_closest_point(pose) - pose for obj in cell.objects], axis=1)
    candidates = [cell.objects[i] for i in range(len(dists)) if dists[i] <= max_dist]
    if len(candidates) < num_mentioned:
        return None

    if select_by == "closest":
        selected_objects = select_objects_closest(candidates, pose, num_mentioned)
    elif select_by == "direction":
        selected_objects = select_objects_direction(candidates, pose, num_mentioned)
    elif select_by == "class":
        selected_objects = select_objects_class(candidates, pose, num_mentioned)
    elif select_by == "random":
        selected_objects = select_objects_random(candidates, pose, num_mentioned)
    elif select_by == "direction_random":
        selected_objects = select_objects_direction_random(candidates, pose, num_mentioned)
    else:
        raise ValueError(f"Invalid selection method: {select_by}.")

    descriptions = []
    for obj in selected_objects:
        if no_ontop:
            direction = get_direction_noOntop(obj, pose)
        else:
            direction = get_direction(obj, pose)
        closest_point = obj.get_closest_point(pose)

        offset_center = pose - obj.get_center()
        offset_closest = pose - closest_point
        # descriptions.append(DescriptionPoseCell(obj.id, obj.instance_id, obj.label, obj.get_color_rgb(), obj.get_color_text(), direction, offset_center, offset_closest, closest_point))
        descriptions.append(
            DescriptionPoseCell(obj, direction, offset_center, offset_closest, closest_point)
        )

    return descriptions


def ground_pose_to_best_cell(
    pose_w: np.ndarray,
    pose_cell_descriptions: List[DescriptionPoseCell],
    cell: Cell,
    all_cells=False,
) -> List[DescriptionBestCell]:
    """Ground a pose to the best-fitting cell in the dataset for performance evaluation.
    There are two types of cells: A "Pose Cell" is created around the pose with the pose in its center
    to form the original description, "Pose Cells" are not part of the actual dataset.
    A "Best Cell" however is the cell of the actual dataset that is closest to the pose.
    It is located without regarding the location of the poses.
    For some evaluations, it is needed to view the pose in the context of the best cell.

    Args:
        pose_w (np.ndarray): Pose in world coordinates
        pose_cell_descriptions (List[DescriptionPoseCell]): Output of describe_pose_in_pose_cell
        cell (Cell): Best cell
        all_cells (bool, optional): The dataset was created using the all_cells option in describe_pose_in_pose_cell. Defaults to False.

    Returns:
        List[DescriptionBestCell]: List of DescriptionBestCell instances
    """

    # Assert cell is valid for this pose
    assert np.all(pose_w >= cell.bbox_w[0:3]) and np.all(
        pose_w <= cell.bbox_w[3:6]
    ), f"{pose_w}, {cell.bbox_w}"

    if all_cells:
        assert len(cell.objects) >= 1, f"Only {len(cell.objects)} objects"
    else:  # Might not be true anymore if all cells are retained
        assert len(cell.objects) >= len(pose_cell_descriptions), f"Only {len(cell.objects)} objects"

    # Norm pose
    pose = (pose_w - cell.bbox_w[0:3]) / cell.cell_size
    assert np.all(pose >= 0) and np.all(pose <= 1.0), f"{pose} {pose_w} {cell.bbox_w}"

    best_cell_descriptions = []

    num_unmatched = 0
    matched_object_ids = []
    # Match the descriptions to the objects in the given cell
    for descr in pose_cell_descriptions:
        # Gather objects that have the correct instance_id and have not been matched yet.
        candidates = [
            obj
            for obj in cell.objects
            if (obj.instance_id == descr.object_instance_id and obj.id not in matched_object_ids)
        ]

        if len(candidates) == 0:  # The description is not matched anymore in the best cell
            best_cell_descriptions.append(DescriptionBestCell.from_unmatched(descr))
            num_unmatched += 1
        else:
            # The description is matched, try to project it to best-matching candidate
            # Select the candidate with the best-matching closest_offset
            # Closest_offset is least likely to have changed
            closest_offsets = np.array(
                [pose - cand.get_closest_point(pose) for cand in candidates]
            )[:, 0:2]
            best_idx = np.argmin(np.linalg.norm(closest_offsets - descr.offset_closest, axis=1))
            best_obj = candidates[best_idx]
            best_closest_offset = closest_offsets[best_idx]

            if (
                np.linalg.norm(descr.offset_closest - best_closest_offset) > np.sqrt(2) / 2
            ):  # Offsets are too different -> not a match. Allow some tolerance here for equally true matches.
                best_cell_descriptions.append(DescriptionBestCell.from_unmatched(descr))
                num_unmatched += 1
            else:  # Offset is close -> object is matched
                matched_object_ids.append(
                    best_obj.id
                )  # Prevent the object from being matched again

                # Calculating these 3 now for best-cell and saving them as well.
                closest_point = best_obj.get_closest_point(pose)
                best_offset_center = pose - best_obj.get_center()
                best_offset_closest = pose - closest_point

                best_cell_descriptions.append(
                    DescriptionBestCell.from_matched(
                        descr, best_obj.id, closest_point, best_offset_center, best_offset_closest
                    )
                )

    return best_cell_descriptions, pose, num_unmatched


def coarse_regenerate_pose_from_gt_cell(
    pose: Pose,
    cell: Cell,
    num_mentioned: int,
    select_by: str,
    all_cells: bool = False,
):
    """Re-build descriptions like prepare.py with describe_best_cell=True: describe in the GT database cell, then ground.

    Returns:
        New Pose, or None if generation failed (caller keeps the original pose).
    """
    location = pose.pose_w
    if not (
        np.all(location >= cell.bbox_w[0:3]) and np.all(location <= cell.bbox_w[3:6])
    ):
        print("Location is not in the cell")
        return None
    try:
        pose_cell_descriptions = describe_pose_in_pose_cell(
            location, cell, select_by, num_mentioned
        )
        if pose_cell_descriptions is None or len(pose_cell_descriptions) < num_mentioned:
            return None
        descriptions, pose_in_cell, _ = ground_pose_to_best_cell(
            location, pose_cell_descriptions, cell, all_cells=all_cells
        )
        if len(descriptions) != num_mentioned:
            return None
        return Pose(
            pose_in_cell,
            location,
            cell.id,
            cell.scene_name,
            descriptions,
            described_by=select_by,
        )
    except (AssertionError, ValueError):
        return None


class Kitti360CoarseDataset(Kitti360BaseDataset):
    def __init__(
        self,
        base_path,
        scene_name,
        transform,
        shuffle_hints=False,
        flip_poses=False,
        sample_close_cell=False,
        hint_type=0,
        add_object_number=False,
        random_choice_direction=False,
        replace_pose=False,
        no_color=False,
        replace_color=False,
        replace_random=False,
        is_global=False,
        constraint_all=False,
        random_discard=False,
        text_mode = 0,
        sub_sample = False,
        transfer_mode=False,
        perfect_matching = False,
        output_detail=False,
        split_type=False,
        inst_num=0,
        change_type=None,
        change_num=0,
        random_generate=False,
        org_cell=False,
        opposite=False,
        regenerate_descriptions=False,
        num_mentioned_coarse=None,
        coarse_regenerate_all_cells=False,
    ):
        """Dataset variant for coarse module training.
        Returns one item per pose.

        Args:
            base_path: Base path of the Kitti360Poses data
            scene_name: Scene name
            transform: PyG transform to apply to object_points
            shuffle_hints (bool, optional): Shuffle the hints of a description. Defaults to False.
            flip_poses (bool, optional): Flip the poses inside the cell. NOTE: Might make hints inaccurate. Defaults to False.
            sample_close_cell (bool, optional): Sample any close-by cell per pose instead of the original one. Defaults to False.
            regenerate_descriptions: If True, replace each pose's descriptions using GT pose_w and
                cells_dict[cell_id] with num_mentioned_coarse (prepare.py describe_best_cell-style).
            num_mentioned_coarse: Number of objects to mention when regenerate_descriptions is True.
            coarse_regenerate_all_cells: Forwarded to ground_pose_to_best_cell (all_cells dataset flag).
        """
        super().__init__(base_path, scene_name, add_object_number, no_color, text_mode, opposite=opposite)
        self.opposite = opposite
        self.shuffle_hints = shuffle_hints
        self.transform = transform
        self.flip_poses = flip_poses
        self.no_color = no_color
        self.replace_color = replace_color
        self.replace_random = replace_random
        self.is_global = is_global
        self.constraint_all = constraint_all
        self.random_discard = random_discard

        self.sample_close_cell = sample_close_cell
        self.cell_centers = np.array([cell.get_center()[0:2] for cell in self.cells])
        self.hint1 = self.hint1_descriptions != []
        self.hint_type = hint_type
        self.add_object_number = add_object_number
        self.random_choice_direction = random_choice_direction
        self.replace_pose = replace_pose
        self.text_mode = text_mode
        self.sub_sample = sub_sample
        self.transfer_mode = transfer_mode
        self.perfect_matching = perfect_matching
        self.output_detail = output_detail
        self.split_type = split_type
        self.aux_limit = 6
        self.base_limit = 6
        self.inst_num = inst_num
        self.random_generate = random_generate
        self.org_cell = org_cell
        self.corrupt_prob = 0.0
        self.corrupt_num = 1

        if random_generate:
            print("Random Generate is True")
        
        if change_type == "none":
            change_type = None
        self.change_type = change_type
        self.change_num = change_num
        assert change_type in ["class", "color", "direction", "discard", None]
        if change_type is not None:
            assert change_num in [1, 2, 3, 4, 5, 6]
            assert not self.shuffle_hints
        else:
            assert change_num == 0

        if sub_sample:
            assert text_mode > 0

        if text_mode == 1:
            assert self.med_texts != []
        if text_mode == 2:
            assert self.hard_texts != []
        if hint_type == 1:
            for hint, hint1, pose in zip(self.hint_descriptions, self.hint1_descriptions, self.poses):
                if pose.information["far"] != -1:
                    hint.extend(hint1[-2:])

        self._maybe_regenerate_coarse_descriptions(
            regenerate_descriptions,
            num_mentioned_coarse,
            coarse_regenerate_all_cells,
        )

    def _maybe_regenerate_coarse_descriptions(
        self,
        regenerate_descriptions: bool,
        num_mentioned_coarse,
        coarse_regenerate_all_cells: bool,
    ):
        if not regenerate_descriptions:
            return
        assert num_mentioned_coarse is not None and num_mentioned_coarse >= 1
        had_information = len(self.poses) > 0 and hasattr(self.poses[0], "information")
        new_poses = []
        n_fail = 0
        n_no_finished = 0

        for pose in tqdm.tqdm(self.poses, desc="Regenerating coarse descriptions"):
            cell = self.cells_dict[pose.cell_id]
            select_by = pose.described_by if getattr(pose, "described_by", None) else "closest"
            if len(cell.objects) < 6:
                repl = None
            else:
                if len(cell.objects) < num_mentioned_coarse:    
                    num_mentioned_input = len(cell.objects)
                else:
                    num_mentioned_input = num_mentioned_coarse
                repl = coarse_regenerate_pose_from_gt_cell(
                    pose,
                    cell,
                    num_mentioned_input,
                    select_by,
                    all_cells=coarse_regenerate_all_cells,
                )
            if repl is None:
                new_poses.append(pose)
                n_fail += 1
            else:
                if num_mentioned_input != num_mentioned_coarse:
                    n_no_finished += 1
                repl = merge_repl_descriptions_with_pose_by_object_id(pose, repl)
                new_poses.append(repl)
                # if not poses_equivalent(pose, repl):
                #     import pdb; pdb.set_trace()
            
        self.poses = new_poses
        if had_information:
            self.poses = add_pose_information(self.poses)
        self.hint_descriptions, self.hint_index = zip(
            *[
                Kitti360BaseDataset.create_hint_description(
                    p, self.cells_dict[p.cell_id], self.no_color, opposite=self.opposite
                )
                for p in self.poses
            ]
        )
        if had_information:
            self.hint1_descriptions, self.hint1_number, self.hint1_index = zip(
                *[
                    Kitti360BaseDataset.create_hint1_description(
                        p, self.add_object_number, True, self.no_color
                    )
                    for p in self.poses
                ]
            )
        print(
            f"[{self.scene_name}] coarse description regeneration: "
            f"{len(self.poses) - n_fail - n_no_finished} ok, {n_fail} kept original (num_mentioned={num_mentioned_coarse}), {n_no_finished} no finished "
        )

    def __getitem__(self, idx):
        pose_org = self.poses[idx]

        replace_pose = self.replace_pose

        if replace_pose:
            replace_pose =  np.random.choice(np.arange(2), size=1, replace=False)[0]

        if replace_pose:
            location = pose_org.pose_w
            best_cell = self.cells_dict[pose_org.cell_id]
            if self.random_choice_direction:
                description_method = "direction_random"
            else:
                description_method = "random"
            num_mentioned = len(pose_org.descriptions)
            descriptions = describe_pose_in_pose_cell(
                    location, best_cell, description_method, num_mentioned
                )
            if descriptions is None or len(descriptions) < 6:
                pose = pose_org
                replace_pose = False
            else:
                descriptions, pose_in_cell, num_unmatched = ground_pose_to_best_cell(
                        location, descriptions, best_cell
                    )
                pose = Pose(
                        pose_in_cell,
                        location,
                        best_cell.id,
                        best_cell.scene_name,
                        descriptions,
                        described_by=description_method,
                    )
                if self.hint1:
                    pose = add_pose_information([pose])[0]
        else:
            pose = pose_org

        # TODO/NOTE: If it doesn't work, check if there is a problem with flipping later on
        if self.sample_close_cell:
            cell_size = self.cells[0].cell_size
            dists = np.linalg.norm(self.cell_centers - pose.pose_w[0:2], axis=1)
            indices = np.argwhere(dists <= cell_size / 2).flatten()
            cell = self.cells[np.random.choice(indices)]
            assert np.linalg.norm(cell.get_center()[0:2] - pose.pose_w[0:2]) < cell_size
        else:
            cell = self.cells_dict[pose.cell_id]
        
        if replace_pose:
            hints, index = Kitti360BaseDataset.create_hint_description(pose, cell)
            hints1,  hints1_num, index1 = Kitti360BaseDataset.create_hint1_description(pose, self.add_object_number, True, self.no_color)

        else:
            hints = self.hint_descriptions[idx]
            index = self.hint_index[idx]
            if self.hint1 and self.text_mode == 0: 
                if self.shuffle_hints:
                    hints1, _, index1 =  Kitti360BaseDataset.create_hint1_description(pose, self.add_object_number, True, self.no_color)
                else:
                    hints1 = self.hint1_descriptions[idx]
                    index1 = self.hint1_index[idx]
                hints1_num = self.hint1_number[idx]
            elif self.text_mode == 1:

                if not self.random_generate:
                    hints1 = self.med_texts[idx]
                    hints1_length = []
                    hints1_label = self.med_texts_label[idx]
                    for hint in hints1:
                        hints1_length += hint.split(". ")
                else:
                    hints1, hints1_label =  generate_moderate_text(self.poses[idx])
                    hints1_length = []
                    for hint in hints1:
                        hints1_length += hint.split(". ")
                index1 = index.copy()
            elif self.text_mode == 2 or self.text_mode == 3 or self.text_mode == 4:
                hints_all = self.hard_texts[idx].replace("\u2019", "'").replace("\u2014", " - ")
                hints1 = [text.replace(".", "") + "." for text in text_tokenize.sent_tokenize(hints_all)]
                index1 = index.copy()

            else:
                hints1_num = []
                index1 = None


        add_des = self.hint_type == 1
        org_des = self.hint_type == -1

        if self.change_type is not None:
            change_idx_list = np.random.choice(np.arange(len(hints)), size=self.change_num, replace=False)
            for change_idx in change_idx_list:
                if self.change_type == "color":
                    original = pose.descriptions[change_idx].object_color_text
                    potential_list = [ele for ele in robust_color_list if ele != original]
                elif self.change_type == "direction":
                    original = pose.descriptions[change_idx].direction
                    potential_list = [ele for ele in robust_direction_list if ele != original]
                elif self.change_type == "class":
                    original = pose.descriptions[change_idx].object_label
                    potential_list = [ele for ele in robust_class_list if ele != original]
                elif self.change_type == "discard":
                    break
                update = np.random.choice(potential_list, size = 1)[0]
                temp = hints[change_idx]
                hints[change_idx] = temp.replace(" " + original + " ", " " + update + " ").replace(" " + original + ".", " " + update + ".")
            
            if self.change_type == "discard":
                new_hints = []
                for change_idx in np.arange(len(hints)):
                    if change_idx not in change_idx_list:
                        new_hints.append(hints[change_idx])
                
                hints = new_hints
            

        # if add_des:
        #     assert self.hint1_descriptions != []
        #     if pose.information["far"] != -1:
        #         hints.extend(hints1[-2:])
        
        # TODO: index shuffle is error here for complex situation, need to fix
        if self.sub_sample:
            if self.text_mode == 1:
                subset_num = np.random.choice(np.arange(1, len(hints1) + 1), size=1, replace=False)[0]
                subset_index = np.random.choice(np.arange(len(hints1)), size=subset_num, replace=False).tolist()
                subset_text = [hints1[sub_idx] for sub_idx in subset_index]
                # subset_raw_index = [hints1_label[sub_idx] for sub_idx in subset_index]
                subset_raw_index = []
                for sub_idx in subset_index:
                    subset_raw_index += hints1_label[sub_idx]

                subset_raw = []
                for subset_raw_idx in subset_raw_index:
                    subset_raw.append(hints[subset_raw_idx])

                subset_text = " ".join(subset_text)
                subset_raw = " ".join(subset_raw)
            elif self.text_mode == 2:
                potential = self.hard_texts_help[idx]
                potential_ready = []
                if self.aux_limit < 6:
                    potential_ready = [potential_i for potential_i in potential if len(potential_i["index"]) <= self.aux_limit]

                elif self.aux_limit > 6:
                    potential_ready = [potential_i for potential_i in potential if len(potential_i["index"]) >= self.aux_limit - self.base_limit]

                if potential_ready != []:
                    potential = potential_ready
                    # else:
                    #     print("Empty!")
                    

                subset_ele = np.random.choice(potential, size=1, replace=False)[0]
                subset_text = subset_ele["texts_hard"].replace("\u2019", "'").replace("\u2014", " - ")
                if self.transfer_mode:
                    subset_raw = subset_ele["texts"]
                else:
                    subset_index = subset_ele["index"]
                    subset_raw = (np.array(hints)[np.array(subset_index)]).tolist()
                    subset_raw = " ".join(subset_raw)

        else:
            subset_text = None
            subset_raw = None

        
        if self.output_detail:
            # colors = [np.argmin(np.linalg.norm(pose_org_i.object_color_rgb - COLORS, axis = 1)) for pose_org_i in pose_org.descriptions]
            if self.text_mode == 2:
                detail_list = subset_ele["gt"]
                detail_index = subset_index

                if self.shuffle_hints: # TODO: Do we really ned this?
                    shuffle_aux_index = np.random.choice(np.arange(len(detail_index)), size=len(detail_index), replace=False)
                    detail_list = (np.array(detail_list)[shuffle_aux_index]).tolist()
                    detail_index = (np.array(detail_index)[shuffle_aux_index]).tolist()

                if self.split_type == "all":
                    detail = torch.zeros(len(detail_list) , 3, dtype = torch.int64)
                    for item_idx, item in enumerate(detail_list):
                        detail[item_idx, 0] = DIRECTION_TO_INDEX[item[0]]
                        detail[item_idx, 1] = COLOR_TO_INDEX[item[1]] # colors[detail_index[item_idx]]
                        detail[item_idx, 2] = CLASS_TO_INDEX[item[2]]

                    detail_out = [{"direction": detail[:,0], "color": detail[:, 1], "class": detail[:, 2]}]

                elif self.split_type == "sentence":
                    pass

        else:
            detail_out = []

        if self.inst_num > 0:
            cell_idx = [des.object_id if des.is_matched else None for des in pose.descriptions]

        else:
            inst_t = None
            inst_s = None

        if self.shuffle_hints:
            shuffle_index = np.random.choice(np.arange(len(hints)), size=len(hints), replace=False)
            hints = (np.array(hints)[shuffle_index]).tolist()
            index = (np.array(index)[shuffle_index]).tolist()
                                  # np.random.choice(hints, size=len(hints), replace=False)
            if self.inst_num > 0:
                cell_idx = (np.array(cell_idx)[shuffle_index]).tolist()

            if self.hint1:
                shuffle_index1 = np.random.choice(np.arange(len(hints1)), size=len(hints1), replace=False)
                hints1 = (np.array(hints1)[shuffle_index1]).tolist()
                if self.text_mode == 0:
                    hints1_num = (np.array(hints1_num)[shuffle_index1]).tolist()
                # index = (np.array(index)[shuffle_index]).tolist()

        if self.inst_num > 0:

            useful_idx = [text_i for text_i in range(len(pose.descriptions)) if cell_idx[text_i] != None]
            assert len(useful_idx) > 0
            if len(useful_idx) >= self.inst_num:
                replace = False
            else:
                replace = True
            choice_idx = np.random.choice(useful_idx, size=self.inst_num, replace=replace)
            inst_t = choice_idx.tolist()
            inst_s = [cell_idx[choice_i] for choice_i in choice_idx]


        if self.text_mode == 0:
            if self.hint1:
                if add_des or org_des:
                    length = len(hints)
                else:
                    length = len(hints1)
                text1 = " ".join(hints1)
            else:
                text1 = ""
                if add_des:
                    length = len(hints)
                else:
                    length = len(pose.descriptions)
        elif self.text_mode == 1:
            length = len(hints1_length)
            hints1_num = [1] * length
            text1 = " ".join(hints1)
        elif self.text_mode == 2 or self.text_mode == 3 or self.text_mode == 4:
            length = len(hints1)
            hints1_num = [1] * length
            text1 = " ".join(hints1)
        else:
            raise NameError


        text = " ".join(hints)

        # NOTE: hints are currently not flipped! (Only the text.)
        if self.flip_poses:
            if np.random.choice((True, False)):  # Horizontal

                pose, cell, text, text1, index, index1 = flip_pose_in_cell(pose, cell, text, text1, index, index1, 1, no_color = self.no_color)

                if self.sub_sample:
                    subset_text = flip_pose_in_cell_easy(subset_text, 1)
                    subset_raw = flip_pose_in_cell_easy(subset_raw, 1)
                    for detail in detail_out:
                        detail["direction"][detail["direction"] == DIRECTION_TO_INDEX["east"]] = -1
                        detail["direction"][detail["direction"] == DIRECTION_TO_INDEX["west"]] = DIRECTION_TO_INDEX["east"]
                        detail["direction"][detail["direction"] == -1] = DIRECTION_TO_INDEX["west"]

            if np.random.choice((True, False)):  # Vertical

                pose, cell, text, text1, index, index1 = flip_pose_in_cell(pose, cell, text, text1, index, index1, -1, no_color = self.no_color)
                if self.sub_sample:
                    subset_text = flip_pose_in_cell_easy(subset_text, -1)
                    subset_raw =flip_pose_in_cell_easy(subset_raw, -1)

                    for detail in detail_out:
                        detail["direction"][detail["direction"] == DIRECTION_TO_INDEX["north"]] = -1
                        detail["direction"][detail["direction"] == DIRECTION_TO_INDEX["south"]] = DIRECTION_TO_INDEX["north"]
                        detail["direction"][detail["direction"] == -1] = DIRECTION_TO_INDEX["south"]

        if self.perfect_matching or self.random_discard:
            index_list = np.array([True] * len(cell.objects))
            label_list = [obj.label for obj in cell.objects]
            keep_index_list = []
            for desc_idx, desc in enumerate(pose.descriptions):
                if desc.is_matched:
                    for obj_ind, obj in enumerate(cell.objects):
                        if obj.id == desc.object_id:
                            keep_index_list.append(obj_ind)
                            break

            discard_list = np.array([obj_ind for obj_ind in range(len(cell.objects)) if obj_ind not in keep_index_list])
            alpha = 1.0 # random discard prob
            beta = 0.5 # random discard for certain index prob
            bionominal = False

            if len(discard_list) != 0:
                if (self.random_discard and np.random.choice([True, False], 1, replace=False, p=[alpha, 1-alpha])[0]) or self.perfect_matching:
                    if bionominal:
                        for dis_index in discard_list:
                            if beta != 1.0:
                                if self.perfect_matching or np.random.choice([True, False], 1, replace=False, p=[beta, 1-beta])[0]:
                                    index_list[dis_index] = False
                            else:
                                if self.perfect_matching or np.random.choice([True, False], 1, replace=False, p=[CLASS_TO_PROB[dis_index], 1-CLASS_TO_PROB[dis_index]])[0]:
                                    index_list[dis_index] = False
                    else: # uniform

                        discard_number = np.random.choice(np.arange(len(discard_list) + 1), 1, replace=False,)[0]
                        if discard_number != 0:
                            real_discard_list = np.random.choice(discard_list, size = discard_number, replace = False)
                            for dis_index in real_discard_list:
                                index_list[dis_index] = False
                                
                cell_objects = []
                for ind, keep in enumerate(index_list):
                    if keep:
                        cell_objects.append(cell.objects[ind])
            else:
                cell_objects = cell.objects


            if inst_s is not None:
                new_inst_s = [100] * len(inst_s)
                for obj_ind, obj in enumerate(cell_objects):
                    if obj.id in inst_s:
                        new_inst_s[inst_s.index(obj.id)] = obj_ind

                inst_s = new_inst_s
        else:
            cell_objects = cell.objects
            
            if inst_s is not None:
                new_inst_s = [100] * len(inst_s)
                for obj_ind, obj in enumerate(cell.objects):
                    if obj.id in inst_s:
                        new_inst_s[inst_s.index(obj.id)] = obj_ind

                inst_s = new_inst_s

        object_points = batch_object_points(cell_objects, self.transform, replace_color=self.replace_color, replace_random=self.replace_random,
                                            constraint_all=self.constraint_all)
        if self.is_global:
            x = object_points.x
            pos = object_points.pos
            ind = torch.zeros(len(x))
            for i in range(len(object_points.ptr)):
                ind[i:(i+1)] = i
            ind = ind / len(x)
            x = torch.cat([x, ind.unsqueeze(-1)], dim = -1)
            global_points = Batch.from_data_list([Data(x=x, pos=pos)])
        else:
            global_points = None

        object_class_indices = [CLASS_TO_INDEX[obj.label] for obj in cell_objects]
        object_color_indices = [COLOR_TO_INDEX[obj.get_color_text()] for obj in cell_objects]
        index = torch.FloatTensor(index)

        if self.org_cell and self.random_discard:
            org_cell_objects = cell.objects
            org_object_points = batch_object_points(org_cell_objects, self.transform, replace_color=self.replace_color, replace_random=self.replace_random,
                                            constraint_all=self.constraint_all)
        else:
            org_cell_objects = None
            org_object_points = None

        corrupt_text = ""
        corrupt_text1 = ""
        corrupt_length = 0

        if self.corrupt_prob > 0 and np.random.random() < self.corrupt_prob:
            corrupt_hints = list(hints)  # shallow copy
            corrupt_types = ["color", "class", "direction"]
            corrupt_idx_list = np.random.choice(
                np.arange(len(corrupt_hints)),
                size=min(self.corrupt_num, len(corrupt_hints)),
                replace=False
            )
            for ci in corrupt_idx_list:
                ctype = np.random.choice(corrupt_types)
                if ctype == "color":
                    original = pose.descriptions[ci].object_color_text
                    candidates = [c for c in robust_color_list if c != original]
                elif ctype == "class":
                    original = pose.descriptions[ci].object_label
                    candidates = [c for c in robust_class_list if c != original]
                else:
                    original = pose.descriptions[ci].direction
                    candidates = [c for c in robust_direction_list if c != original]
                replacement = np.random.choice(candidates)
                corrupt_hints[ci] = corrupt_hints[ci].replace(
                    " " + original + " ", " " + replacement + " "
                ).replace(
                    " " + original + ".", " " + replacement + "."
                )
            corrupt_text = " ".join(corrupt_hints)
            corrupt_text1 = corrupt_text
            corrupt_length = length


        return {
            "poses": pose,
            "cells": cell,
            "objects": cell_objects,
            "object_points": object_points,
            "global_points": global_points,
            "org_objects": org_cell_objects,
            "org_object_points": org_object_points,
            "texts": text,
            "texts1": text1,
            "sub_text": subset_text,
            "sub_gt": subset_raw,
            "sub_target": detail_out,
            "length": length,
            "texts1_number": hints1_num,
            "cell_ids": pose.cell_id,
            "scene_names": self.scene_name,
            "object_class_indices": object_class_indices,
            "object_color_indices": object_color_indices,
            # "text_class_indices": text_class_indices,
            # "text_color_indices": text_color_indices,
            # "text_direction_indices": text_direction_indices,
            "debug_hint_descriptions": hints,  # Care: Not shuffled etc!
            "index": index,
            "inst_s": inst_s,
            "inst_t": inst_t,
            "corrupt_texts": corrupt_text,
            "corrupt_texts1": corrupt_text1,
            "corrupt_length": corrupt_length,
        }

    def __len__(self):
        return len(self.poses)
    
    def update_limit(self, num):
        self.aux_limit = num


class Kitti360CoarseDatasetMulti(Dataset):
    def __init__(
        self,
        base_path,
        scene_names,
        transform,
        shuffle_hints=False,
        flip_poses=False,
        sample_close_cell=False,
        hint_type=0,
        add_object_number=False,
        random_choice_direction=False,
        replace_pose=False,
        no_color=False,
        replace_color=False,
        replace_random=False,
        is_global=False,
        constraint_all=False,
        random_discard=False,
        text_mode=0,
        sub_sampie=False,
        transfer_mode=False,
        perfect_matching=False,
        output_detail=False,
        split_type=False,
        inst_num=False,
        change_type=None,
        change_num=0,
        random_generate=False,
        org_cell=False,
        opposite=False,
        regenerate_descriptions=False,
        num_mentioned_coarse=None,
        coarse_regenerate_all_cells=False,
    ):
        """Multi-scene variant of Kitti360CoarseDataset.

        Args:
            base_path: Base path of the Kitti360Poses data
            scene_names: List of scene names
            transform: PyG transform to apply to object_points
            shuffle_hints (bool, optional): Shuffle the hints of a description. Defaults to False.
            flip_poses (bool, optional): Flip the poses inside the cell. NOTE: Might make hints inaccurate. Defaults to False.
            sample_close_cell (bool, optional): Sample any close-by cell per pose instead of the original one. Defaults to False.
        """
        self.scene_names = scene_names
        self.transform = transform
        self.flip_poses = flip_poses
        self.sample_close_cell = sample_close_cell
        self.is_global = is_global
        self.replace_color = replace_color
        self.aux_limit = 6
        self.datasets = [
            Kitti360CoarseDataset(
                base_path, scene_name, transform, shuffle_hints, flip_poses, sample_close_cell, hint_type, 
                add_object_number, random_choice_direction, replace_pose, no_color, replace_color, replace_random, is_global,
                constraint_all, random_discard, text_mode, sub_sampie, transfer_mode, perfect_matching, output_detail, split_type,
                inst_num, change_type, change_num, random_generate, org_cell, opposite,
                regenerate_descriptions=regenerate_descriptions,
                num_mentioned_coarse=num_mentioned_coarse,
                coarse_regenerate_all_cells=coarse_regenerate_all_cells,
            )
            for scene_name in scene_names
        ]

        self.all_cells = [
            cell for dataset in self.datasets for cell in dataset.cells
        ]  # For cell-only dataset
        self.all_poses = [pose for dataset in self.datasets for pose in dataset.poses]  # For eval

        cell_ids = [cell.id for cell in self.all_cells]
        assert len(np.unique(cell_ids)) == len(self.all_cells)  # IDs should not repeat

        print(str(self))

    def __getitem__(self, idx):
        count = 0
        for dataset in self.datasets:
            idx_in_dataset = idx - count
            if idx_in_dataset < len(dataset):
                return dataset[idx_in_dataset]
            else:
                count += len(dataset)
        assert False

    def __len__(self):
        return np.sum([len(ds) for ds in self.datasets])

    def __repr__(self):
        poses = np.array([pose.pose_w for pose in self.all_poses])
        num_poses = len(
            np.unique(poses, axis=0)
        )  # CARE: Might be possible that is is slightly inaccurate if there are actually overlaps
        return f"Kitti360CellDatasetMulti: {len(self.scene_names)} scenes, {len(self)} descriptions for {num_poses} unique poses, {len(self.all_cells)} cells, flip {self.flip_poses}, close-cell {self.sample_close_cell}"

    # def get_known_words(self):
    #     known_words = []
    #     for ds in self.datasets:
    #         known_words.extend(ds.get_known_words())
    #     return list(np.unique(known_words))

    def get_known_classes(self):
        known_classes = []
        for ds in self.datasets:
            known_classes.extend(ds.get_known_classes())
        return list(np.unique(known_classes))

    def get_cell_dataset(self):
        return Kitti360CoarseCellOnlyDataset(self.all_cells, self.transform, self.is_global, self.replace_color)
    
    def update_limit(self, num):
        self.aux_limit = num
        for dataset in self.datasets:
            dataset.update_limit(num)


class Kitti360CoarseCellOnlyDataset(Dataset):
    def __init__(self, cells: List[Cell], transform, is_global = False, replace_color = False):
        """Dataset to return only the cells for encoding during evaluation
        NOTE: The way the cells are read from the Cells-Only-Dataset, they may have been augmented differently during the actual training. Cells-Only does not flip and shuffle!
        """
        super().__init__()

        self.cells = cells
        self.transform = transform
        self.is_global = is_global
        self.replace_color = replace_color
        self.multiple = 0

    def __getitem__(self, idx):
        cell = self.cells[idx]
        assert len(cell.objects) >= 1
        object_points = batch_object_points(cell.objects, self.transform, replace_color=self.replace_color) #, replace_random=self.replace_random)
        if self.multiple != 0:

            object_points_list = []
            objects_list = []
            for _ in range(self.multiple):
                cell_objects = []
                while len(cell_objects) == 0:
                    cell_objects = [obj for obj in cell.objects if np.random.rand() > 0.25]
                mul_object_points = batch_object_points(cell_objects, self.transform, replace_color=self.replace_color)
                object_points_list.append(mul_object_points)
                objects_list.append(cell_objects)

        else:
            object_points_list = None
            objects_list = None


        if self.is_global:
            x = object_points.x
            pos = object_points.pos
            ind = torch.zeros(len(x))
            for i in range(len(object_points.ptr)):
                ind[i:(i+1)] = i
            ind = ind / len(x)
            x = torch.cat([x, ind.unsqueeze(-1)], dim = -1)
            global_points = Batch.from_data_list([Data(x=x, pos=pos)])
        else:
            global_points = None

        return {
            "cells": cell,
            "cell_ids": cell.id,
            "objects": cell.objects,
            "object_points": object_points,
            "objects_list": objects_list,
            "object_points_list":object_points_list,
            "global_points": global_points,
        }

    def __len__(self):
        return len(self.cells)


if __name__ == "__main__":
    base_path = "./data/k360_30-10_scG_pd10_pc8_spY_all_nm6/"

    transform = T.FixedPoints(256)

    for scene_names in (SCENE_NAMES, SCENE_NAMES_TRAIN, SCENE_NAMES_VAL, SCENE_NAMES_TEST):
        dataset = Kitti360CoarseDatasetMulti(
            base_path, scene_names, transform, shuffle_hints=False, flip_poses=False
        )
        # data = dataset[0]
        # pose, cell, text = data['poses'], data['cells'], data['texts']
        # offsets = np.array([descr.offset_closest for descr in pose.descriptions])
        # hints = text.split('.')
        # pose_f, cell_f, text_f, hints_f, offsets_f = flip_pose_in_cell(pose, cell, text, 1, hints=hints, offsets=offsets)

        # Gather information about duplicate descriptions
        descriptors = []
        for pose in dataset.all_poses:
            mentioned = sorted(
                [f"{d.object_label}_{d.object_color_text}_{d.direction}" for d in pose.descriptions]
            )
            descriptors.append(mentioned)

        unique, counts = np.unique(descriptors, return_counts=True)
        # for d in descriptors[0:10]:
        #     print('\t',d)
        print(
            f"{len(descriptors)} poses, {len(unique)} uniques, {np.max(counts)} max duplicates, {np.mean(counts):0.2f} mean duplicates"
        )
        print("---- \n\n")