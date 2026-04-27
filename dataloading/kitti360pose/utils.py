from typing import List
import numpy as np
from datapreparation.kitti360pose.imports import Object3d, Cell, Pose
from copy import deepcopy

import torch
from torch_geometric.data import Data, Batch
import torch_geometric.transforms as T

from configs.mixed0 import CLASS_TO_COLOR, COLOR_NAMES, CLASS_TO_INDEX


OBJECT_LIST = ["building", "pole", "light", "sign", "garage", "stop", "smallpole", "lamp", "bin", "machine", "box"]
# TODO: for free orientations, possibly flip cell only, create descriptions and hints again
# OR: numeric vectors in descriptions, flip cell objects and description.direction, then create hints again
# Flip pose, too?
def flip_pose_in_cell_easy(text, direction):
    if direction == 1:  # Horizontally
        text = (
            text.replace("east", "east-flipped")
            .replace("west", "east")
            .replace("east-flipped", "west")
        )

    elif direction == -1:
            text = (
            text.replace("north", "north-flipped")
            .replace("south", "north")
            .replace("north-flipped", "south")
        )
    
    return text


def flip_pose_in_cell(pose: Pose, cell: Cell, text, text1, index, index1, direction, hints=None, offsets=None, no_color = False):
    """Flips the cell horizontally or vertically
    CARE: Needs adjustment for non-compass directions
    CARE: Description.object_closest_point is flipped but direction in description is not flipped.

    Args:
        pose (Pose): The pose to flip, is copied before modification
        cell (Cell): The cell to flip, is copied before modification
        text (str): The text description to flip
        direction (int): Horizontally (+1) or vertically (-1)

    Returns:
        Pose: flipped pose
        Cell: flipped cell
        str: flipped text
    """
    assert direction in (-1, 1)
    assert sum([hints is None, offsets is None]) != 1  # Either both or none

    pose = deepcopy(pose)
    cell = deepcopy(cell)
    if offsets is not None:
        offsets = offsets.copy()

    if direction == 1:  # Horizontally
        pose.pose[0] = 1.0 - pose.pose[0]
        for obj in cell.objects:
            obj.xyz[:, 0] = 1 - obj.xyz[:, 0]
        for descr in pose.descriptions:
            descr.closest_point[0] = 1.0 - descr.closest_point[0]

        text = (
            text.replace("east", "east-flipped")
            .replace("west", "east")
            .replace("east-flipped", "west")
        ) 

        text1 = (
            text1.replace("east", "east-flipped")
            .replace("west", "east")
            .replace("east-flipped", "west")
        )                         

        if hints is not None:
            hints = [
                hint.replace("east", "east-flipped")
                .replace("west", "east")
                .replace("east-flipped", "west")
                for hint in hints
            ]
            offsets[:, 0] *= -1
        
        if no_color:
            if index is not None:
                for id in range(len(index)):
                    if index[id][0] == 2:
                        index[id] = (3, index[id][1])
                    elif index[id][0] == 3:
                        index[id] = (2, index[id][1])

            if index1 is not None:
                for id in range(len(index1)):
                    if index1[id][0] == 2:
                        index1[id] = (3, index1[id][1])
                    elif index[id][0] == 3:
                        index1[id] = (2, index1[id][1])

        else:
            if index is not None:
                for id in range(len(index)):
                    if index[id][0] == 2:
                        index[id] = (3, index[id][1], index[id][2])
                    elif index[id][0] == 3:
                        index[id] = (2, index[id][1], index[id][2])

            if index1 is not None:
                for id in range(len(index1)):
                    if index1[id][0] == 2:
                        index1[id] = (3, index1[id][1], index1[id][2])
                    elif index[id][0] == 3:
                        index1[id] = (2, index1[id][1], index1[id][2])


    elif direction == -1:  # Vertically
        pose.pose[1] = 1.0 - pose.pose[1]
        for obj in cell.objects:
            obj.xyz[:, 1] = 1 - obj.xyz[:, 1]
        for descr in pose.descriptions:
            descr.closest_point[1] = 1.0 - descr.closest_point[1]

        text = (
            text.replace("north", "north-flipped")
            .replace("south", "north")
            .replace("north-flipped", "south")
        )

        text1 = (
            text1.replace("east", "east-flipped")
            .replace("west", "east")
            .replace("east-flipped", "west")
        )       
                    
        if hints is not None:
            hints = [
                hint.replace("north", "north-flipped")
                .replace("south", "north")
                .replace("north-flipped", "south")
                for hint in hints
            ]
            offsets[:, 1] *= -1
        
        if no_color:

            if index is not None:
                for id in range(len(index)):
                    if index[id][0] == 0:
                        index[id] = (1, index[id][1])
                    elif index[id][0] == 1:
                        index[id] = (0, index[id][1])

            if index1 is not None:
                for id in range(len(index1)):
                    if index1[id][0] == 0:
                        index1[id] = (1, index1[id][1])
                    elif index[id][0] == 1:
                        index1[id] = (0, index1[id][1])
        else:

            if index is not None:
                for id in range(len(index)):
                    if index[id][0] == 0:
                        index[id] = (1, index[id][1], index[id][2])
                    elif index[id][0] == 1:
                        index[id] = (0, index[id][1], index[id][2])

            if index1 is not None:
                for id in range(len(index1)):
                    if index1[id][0] == 0:
                        index1[id] = (1, index1[id][1], index1[id][2])
                    elif index[id][0] == 1:
                        index1[id] = (0, index1[id][1], index1[id][2])

    assert "flipped" not in text

    if hints is not None:
        return pose, cell, text, text1, index, index1, hints, offsets
    else:
        return pose, cell, text, text1, index, index1


def batch_object_points(objects: List[Object3d], transform, use_fps=False, use_gather=False, use_knn=False,
                        keep_num=0, replace_color = False, replace_random = False, constraint_all=False):
    """Generates a PyG-Batch for the objects of a single cell.
    Note: Aggregating an entire batch of cells into a single PyG-Batch would exceed the limit of 256 sub-graphs.
    Note: The objects can be transformed / augmented freely, as their center-points are encoded separately.

    Args:
        objects (List[Object3d]): Cell objects
        transform: PyG-Transform
    """

    # CARE: Transforms not working with batches?! Doing it object-by-object here!
    if use_knn:
        assert use_fps
        rgb = [torch.tensor(obj.rgb, dtype=torch.float).cuda() for obj in objects]
        xyz = [torch.tensor(obj.xyz, dtype=torch.float).cuda() for obj in objects]


        if use_gather:
            xyz, rgb = zip(*[fps(xyz_i.unsqueeze(0), keep_num, rgb_i.unsqueeze(0)) for xyz_i, rgb_i in zip (xyz, rgb)])
        else:
            xyz, rgb = zip(*[fps2(xyz_i.unsqueeze(0), keep_num, rgb_i.unsqueeze(0)) for xyz_i, rgb_i in zip (xyz, rgb)])

        return torch.cat(xyz)

    else:
        
        if use_fps:
            rgb = [torch.tensor(obj.rgb, dtype=torch.float).cuda() for obj in objects]
            xyz = [torch.tensor(obj.xyz, dtype=torch.float).cuda() for obj in objects]


            if use_gather:
                xyz, rgb = zip(*[fps(xyz_i.unsqueeze(0), keep_num, rgb_i.unsqueeze(0)) for xyz_i, rgb_i in zip (xyz, rgb)])
            else:
                xyz, rgb = zip(*[fps2(xyz_i.unsqueeze(0), keep_num, rgb_i.unsqueeze(0)) for xyz_i, rgb_i in zip (xyz, rgb)])

            data_list = [
                Data(
                    x=rgb_i.squeeze(0), pos=xyz_i.squeeze(0)
                )
                for rgb_i, xyz_i in zip(rgb, xyz)
            ]

        else:
            if replace_color:
                data_list = [
                    Data(
                        x=torch.tensor(CLASS_TO_COLOR[obj.label], dtype=torch.float).unsqueeze(0).repeat(len(obj.rgb), 1) / 255, pos=torch.tensor(obj.xyz, dtype=torch.float)
                    )
                    for obj in objects
                ]

            else:
                if constraint_all:
                    data_list = [
                            Data(
                                x=torch.tensor(obj.rgb, dtype=torch.float), pos=torch.tensor(obj.xyz, dtype=torch.float),
                                y=torch.tensor([CLASS_TO_INDEX[obj.label]], dtype=torch.float),
                                y_color=torch.tensor([COLOR_NAMES.index(obj.get_color_text())], dtype=torch.float),
                            )
                            for obj in objects
                        ]
                else:
                    data_list = [
                        Data(
                            x=torch.tensor(obj.rgb, dtype=torch.float), pos=torch.tensor(obj.xyz, dtype=torch.float),

                        )
                        for obj in objects
                    ]
        
        for i in range(len(data_list)):
            data_list[i] = transform(data_list[i])

        assert len(data_list) >= 1
        batch = Batch.from_data_list(data_list)
    return batch


def batch_global_points(objects: List[Object3d], transform, use_fps=False, use_gather=False, use_knn=False, keep_num=0, 
                        replace_color = False, replace_random=False):
    """Generates a PyG-Batch for the objects of a single cell.
    Note: Aggregating an entire batch of cells into a single PyG-Batch would exceed the limit of 256 sub-graphs.
    Note: The objects can be transformed / augmented freely, as their center-points are encoded separately.

    Args:
        objects (List[Object3d]): Cell objects
        transform: PyG-Transform
    """

    # CARE: Transforms not working with batches?! Doing it object-by-object here!
    if use_knn:
        # assert use_fps
        # rgb = [torch.tensor(obj.rgb, dtype=torch.float).cuda() for obj in objects]
        # xyz = [torch.tensor(obj.xyz, dtype=torch.float).cuda() for obj in objects]


        # if use_gather:
        #     xyz, rgb = zip(*[fps(xyz_i.unsqueeze(0), keep_num, rgb_i.unsqueeze(0)) for xyz_i, rgb_i in zip (xyz, rgb)])
        # else:
        #     xyz, rgb = zip(*[fps2(xyz_i.unsqueeze(0), keep_num, rgb_i.unsqueeze(0)) for xyz_i, rgb_i in zip (xyz, rgb)])

        # return torch.cat(xyz)
        pass

    else:
        
        if use_fps:
            # rgb = [torch.tensor(obj.rgb, dtype=torch.float).cuda() for obj in objects]
            # xyz = [torch.tensor(obj.xyz, dtype=torch.float).cuda() for obj in objects]


            # if use_gather:
            #     xyz, rgb = zip(*[fps(xyz_i.unsqueeze(0), keep_num, rgb_i.unsqueeze(0)) for xyz_i, rgb_i in zip (xyz, rgb)])
            # else:
            #     xyz, rgb = zip(*[fps2(xyz_i.unsqueeze(0), keep_num, rgb_i.unsqueeze(0)) for xyz_i, rgb_i in zip (xyz, rgb)])

            # data_list = [
            #     Data(
            #         x=rgb_i.squeeze(0), pos=xyz_i.squeeze(0)
            #     )
            #     for rgb_i, xyz_i in zip(rgb, xyz)
            # ]

            pass

        else:
            if replace_color:
                x= []
                pos = []
                inds = []
                for ind, obj in enumerate(objects):
                    x.append(torch.tensor(CLASS_TO_COLOR[obj.label], dtype=torch.float).unsqueeze(0).repeat(len(obj.rgb), 1) / 255)
                    pos.append(torch.tensor(obj.xyz, dtype=torch.float))
                    inds.append(torch.zeros(len(obj.xyz), 1, dtype=torch.float) + ind // len(objects))

                x = torch.stack(x)
                pos = torch.stack(pos)
                inds = torch.stack(inds) 
                x = torch.cat([x, inds], dim = -1)

                data = Data(x=x, pos=pos)

            else:
                x= []
                pos = []
                inds = []
                for ind, obj in enumerate(objects):
                    x.append(torch.tensor(obj.rgb, dtype=torch.float))
                    pos.append(torch.tensor(obj.xyz, dtype=torch.float))
                    inds.append(torch.zeros(len(obj.xyz), 1, dtype=torch.float) + ind // len(objects))

                x = torch.cat(x)
                pos = torch.cat(pos)
                inds = torch.cat(inds) 
                x = torch.cat([x, inds], dim = -1)

                data = Data(x=x, pos=pos)
          

        data = transform(data)

        batch = Batch.from_data_list([data])
    return batch