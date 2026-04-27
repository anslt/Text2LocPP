# python -m datapreparation.kitti360pose.update --scene_name 2013_05_28_drive_0000_sync --path_in ./data/k360_30-10_scG_pd10_pc4_spY_all --new --shift_cells --grid_cells
from typing import List
from datapreparation.args import parse_arguments
from datapreparation.kitti360pose.imports import Cell
import os 
import os.path as osp
import pickle
import numpy as np
import json

def add_direction_information(cells: List[Cell]) -> List[Cell]:
    infos = {}
    for cell_idx, cell in enumerate(cells):
        cell_id = cell.id
        information = {
            "east": None,
            "west": None,
            "north": None,
            "south": None,
            "northeast": None,
            "northwest": None,
            "southeast": None,
            "southwest": None,
        }
        loc = cell.get_center()[:2]
        for cell_near in cells:
            cell_near_id = cell_near.id
            lan_flag = -100
            lon_flag = -100
            loc_near = cell_near.get_center()[:2]
            diff = loc_near - loc
            
            if diff[0] == -10:
                lon_flag = -1
            elif diff[0] == 10:
                lon_flag = 1
            elif diff[0] == 0:
                lon_flag = 0
            
            if diff[1] == -10:
                lan_flag = -1
            elif diff[1] == 10:
                lan_flag = 1
            elif diff[1] == 0:
                lan_flag = 0

            if lon_flag == 1 and lan_flag == 0:
                information["east"] = cell_near_id 
            elif lon_flag == -1 and lan_flag == 0:
                information["west"] = cell_near_id 
            elif lon_flag == 1 and lan_flag == 1:
                information["northeast"] = cell_near_id 
            elif lon_flag == -1 and lan_flag == 1:
                information["northwest"] = cell_near_id 
            elif lon_flag == 1 and lan_flag == -1:
                information["southeast"] = cell_near_id 
            elif lon_flag == -1 and lan_flag == -1:
                information["southwest"] = cell_near_id 
            elif lon_flag == 0 and lan_flag == 1:
                information["north"] = cell_near_id 
            elif lon_flag == 0 and lan_flag == -1:
                information["south"] = cell_near_id

        infos[cell_id] = information 
                                                
    return infos


if __name__ == "__main__":
    args = parse_arguments()
    print(str(args).replace(",", "\n"))
    print()

    path_cells = osp.join(args.path_in, "cells", f"{args.scene_name}.pkl")
    path_out = osp.join(args.path_in, "direction", f"{args.scene_name}.json")
    print("Real output folder:", path_out)

    try:
        os.mkdir(osp.join(args.path_in, "direction"))
    except:
        pass

    cells = pickle.load(open(path_cells, "rb"))
    infos = add_direction_information(cells)

    # pickle.dump(infos, open(path_out, "wb"))
    with open(path_out, 'w', encoding='utf-8') as f:
        json.dump(infos, f, ensure_ascii=True, indent=4)


    # python datapreparation.kitti360pose.update