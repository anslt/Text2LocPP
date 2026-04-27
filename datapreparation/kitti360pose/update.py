# python -m datapreparation.kitti360pose.update --scene_name 2013_05_28_drive_0000_sync --path_in ./data/k360_30-10_scG_pd10_pc4_spY_all --new --shift_poses --grid_cells
from typing import List
from datapreparation.args import parse_arguments
from datapreparation.kitti360pose.imports import Pose
import os 
import os.path as osp
import pickle
import numpy as np

def add_pose_information(poses: List[Pose]) -> List[Pose]:
    for pose_id, pose in enumerate(poses):
        information = {"south": [], "north": [], "east": [], "west": [], "on-top": [], "near": -1, "far": -1}
        dis1 = []
        dis2 = []
        for idx, desc in enumerate(pose.descriptions):
            information[desc.direction].append(idx)
            if desc.direction == "on-top":
                dis1.append(10)
                dis2.append(-10)
            else:
                if desc.is_matched:
                    dis = np.linalg.norm(pose.pose[:2] - desc.best_offset_closest)
                else:
                    dis = np.linalg.norm(pose.pose[:2] - desc.offset_closest)
                dis1.append(dis)
                dis2.append(dis)
        dis1 = np.array(dis1)
        dis2 = np.array(dis2)
        small = np.min(dis1)
        if small == 10:
            print("All on-top in No." + str(pose_id))
            pose.information = information
            continue
        small_index = np.argmin(dis1)
        large_index = np.argmax(dis2)
        if small_index == large_index:
            print("Only 1 object not on-top in No." + str(pose_id))
            pose.information = information
            continue
        else:
            information["near"] = small_index
            information["far"] = large_index

        pose.information = information
    
    return poses


if __name__ == "__main__":
    args = parse_arguments()
    print(str(args).replace(",", "\n"))
    print()

    path_poses = osp.join(args.path_in, "poses", f"{args.scene_name}.pkl")
    path_out = osp.join(args.path_in, "poses1", f"{args.scene_name}.pkl")

    try:
        os.mkdir(osp.join(args.path_in, "poses1"))
    except:
        pass

    poses = pickle.load(open(path_poses, "rb"))
    poses = add_pose_information(poses)

    pickle.dump(poses, open(path_out, "wb"))


    # python datapreparation.kitti360pose.update