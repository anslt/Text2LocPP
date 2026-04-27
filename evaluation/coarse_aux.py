from datapreparation.kitti360pose.imports import Object3d
import numpy as np
import os
import os.path as osp
import cv2
from easydict import EasyDict
from copy import deepcopy
import pickle

import torch
from torch.utils.data import DataLoader
import time

from scipy.spatial.distance import cdist

from models.cell_retrieval import CellRetrievalNetwork

from evaluation.args import parse_arguments
from evaluation.utils import calc_sample_accuracies, print_accuracies

from dataloading.kitti360pose.cells import Kitti360CoarseDataset, Kitti360CoarseDatasetMulti
from dataloading.kitti360pose.eval import Kitti360TopKDataset

# from configs.mixed0 import SCENE_NAMES_TEST, SCENE_NAMES_VAL
from configs.mixed0 import COLOR_NAMES as COLOR_NAMES_K360, KNOWN_CLASS

# from training.coarse_aux import eval_epoch as eval_epoch_retrieval
# from training.utils import plot_retrievals
from models.cross_matcher import get_pos_in_cell

import torch_geometric.transforms as T
import tqdm
import json

"""
TODO:
- Try to add num_matches*10 + sum(match_scores[correctly_matched])
"""

@torch.no_grad()
def eval_epoch_retrieval(model, dataloader, args, return_encodings=False, return_distance=False, aux = True, origin = False):
    assert args.ranking_loss != "triplet"  # Else also update evaluation.pipeline

    model.eval()  
    accuracies = {k: [] for k in args.top_k}
    accuracies_close = {k: [] for k in args.top_k}

    cells_dataset = dataloader.dataset.get_cell_dataset()
    cells_dataloader = DataLoader(
        cells_dataset,
        batch_size=1,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )
    cells_dict = {cell.id: cell for cell in cells_dataset.cells}
    cell_size = cells_dataset.cells[0].cell_size

    if args.add_anchor:
        cell_encodings = torch.zeros(len(cells_dataset), args.anchor_size ** 2, model.embed_dim).to(model.device)
    else:
        cell_encodings = torch.zeros(len(cells_dataset), model.embed_dim).to(model.device)
    db_cell_ids = np.zeros(len(cells_dataset), dtype="<U32")

    text_encodings = torch.zeros((len(dataloader.dataset), model.embed_dim)).to(model.device)
    query_cell_ids = np.zeros(len(dataloader.dataset), dtype="<U32")
    query_poses_w = np.array([pose.pose_w[0:2] for pose in dataloader.dataset.all_poses])

    # Encode the query side
    t0 = time.time()
    index_offset = 0

    is_print = True
    for batch in tqdm.tqdm(dataloader):

        # if args.debug:
        #     import pdb; pdb.set_trace()
        if batch["length"][0] != 0 and not origin:
            if args.hint_type == 0:
                lang = batch["texts1"]
            elif args.hint_type == 1 or args.hint_type == -1:
                lang = batch["texts"]
            if aux:
                text_enc, _, _ = model.encode_aux_text(lang, batch["length"], batch["texts1_number"])
            else:
                text_enc, _, _ = model.encode_text(lang, batch["length"], batch["texts1_number"])
            if is_print:
                print(lang)
                is_print = False
        else:
            assert args.hint_type == -1 or origin
            text_enc, _ = model.encode_text(batch["texts"])
            if is_print:
                print(batch["texts"])
                is_print = False
        batch_size = len(text_enc)

        text_encodings[index_offset : index_offset + batch_size, :] = text_enc.detach()
        
        query_cell_ids[index_offset : index_offset + batch_size] = np.array(batch["cell_ids"])
        index_offset += batch_size
    print(f"Encoded {len(text_encodings)} query texts in {time.time() - t0:0.2f}.")

    # Encode the database side
    index_offset = 0
    for batch in tqdm.tqdm(cells_dataloader):

        cell_enc, _, cell_anchor, _ = model.encode_objects(batch["objects"], batch["object_points"], batch["global_points"])
        batch_size = len(cell_enc)

        if cell_anchor is None:

            cell_encodings[index_offset : index_offset + batch_size, :] = cell_enc.detach()

        else:

            cell_encodings[index_offset : index_offset + batch_size, :, :] = cell_enc.detach()
            
        db_cell_ids[index_offset : index_offset + batch_size] = np.array(batch["cell_ids"])
        index_offset += batch_size

    top_retrievals = {}  # {query_idx: top_cell_ids}
    if return_distance:
        dists_list = []
        scores_list = []

    for query_idx in range(len(text_encodings)):
        if args.ranking_loss != "triplet":  # Asserted above
            if cell_anchor is None:
                scores = torch.mm(cell_encodings, text_encodings[query_idx].unsqueeze(1)).squeeze(1).cpu().detach().numpy()
            else:
                N, N_pos, _ = cell_encodings.shape
                scores = torch.mm(cell_encodings.view(N * N_pos, -1), text_encodings[query_idx].unsqueeze(1))
                scores = scores.view(N, N_pos)
                scores = torch.max(scores, axis=1)[0].cpu().detach().numpy()
            assert len(scores) == len(dataloader.dataset.all_cells) 
            sorted_indices = np.argsort(-1.0 * scores)  # High -> low

        sorted_indices = sorted_indices[0 : np.max(args.top_k)]

        # Best-cell hit accuracy
        retrieved_cell_ids = db_cell_ids[sorted_indices]
        target_cell_id = query_cell_ids[query_idx]

        for k in args.top_k:
            accuracies[k].append(target_cell_id in retrieved_cell_ids[0:k])
        top_retrievals[query_idx] = retrieved_cell_ids

        # Close-by accuracy
        # CARE/TODO: can be wrong across scenes!
        target_pose_w = query_poses_w[query_idx]
        retrieved_cell_poses = [
            cells_dict[cell_id].get_center()[0:2] for cell_id in retrieved_cell_ids
        ]
        dists = np.linalg.norm(target_pose_w - retrieved_cell_poses, axis=1)
        if return_distance:
            dists_list.append(dists[0:max(args.top_k)])
            scores_list.append(scores[sorted_indices])
        for k in args.top_k:
            accuracies_close[k].append(np.any(dists[0:k] <= cell_size / 2))

    for k in args.top_k:
        accuracies[k] = np.mean(accuracies[k])
        accuracies_close[k] = np.mean(accuracies_close[k])

    if return_encodings:
        return accuracies, accuracies_close, top_retrievals, cell_encodings, text_encodings
    elif return_distance:
        return accuracies, accuracies_close, top_retrievals, cell_encodings, text_encodings, np.stack(dists_list), np.stack(scores_list)
    else:
        return accuracies, accuracies_close, top_retrievals


@torch.no_grad()
def run_coarse(model, dataloader, args):
    """Run text-to-cell retrieval to obtain the top-cells and coarse pose accuracies.

    Args:
        model: retrieval model
        dataloader: retrieval dataset
        args: global arguments

    Returns:
        [List]: retrievals as [(cell_indices_i_0, cell_indices_i_1, ...), (cell_indices_i+1, ...), ...] with i ∈ [0, len(poses)-1], j ∈ [0, max(top_k)-1]
        [Dict]: accuracies
    """
    model.eval()

    all_cells_dict = {cell.id: cell for cell in dataloader.dataset.all_cells}


    retrieval_accuracies, retrieval_accuracies_close, retrievals = eval_epoch_retrieval(
        model, dataloader, args
    )
    retrievals = [retrievals[idx] for idx in range(len(retrievals))]  # Dict -> list
    print("Retrieval Accs:")
    print(retrieval_accuracies)
    print("Retrieval Accs Close:")
    print(retrieval_accuracies_close)
    assert len(retrievals) == len(dataloader.dataset.all_poses)

    # Gather the accuracies for each sample
    accuracies = {k: {t: [] for t in args.threshs} for k in args.top_k}
    for i_sample in range(len(retrievals)):
        pose = dataloader.dataset.all_poses[i_sample]
        top_cells = [all_cells_dict[cell_id] for cell_id in retrievals[i_sample]]
        pos_in_cells = 0.5 * np.ones((len(top_cells), 2))  # Predict cell-centers
        accs = calc_sample_accuracies(pose, top_cells, pos_in_cells, args.top_k, args.threshs)

        for k in args.top_k:
            for t in args.threshs:
                accuracies[k][t].append(accs[k][t])

    for k in args.top_k:
        for t in args.threshs:
            accuracies[k][t] = np.mean(accuracies[k][t])

    return retrievals, accuracies


if __name__ == "__main__":
    args = parse_arguments()
    print(str(args).replace(",", "\n"), "\n")

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device:", device, torch.cuda.get_device_name(0))


    config_file = json.load(open(os.path.join("configs", args.configs + ".json")))
    SCENE_NAMES = config_file["SCENE_NAMES"]
    SCENE_NAMES_TRAIN = config_file["SCENE_NAMES_TRAIN"]
    SCENE_NAMES_VAL = config_file["SCENE_NAMES_VAL"]
    SCENE_NAMES_TEST = config_file["SCENE_NAMES_TEST"]

    if args.use_test_set:
        assert len(SCENE_NAMES_TEST) > 0


    # Load datasets
    if args.no_pc_augment:
        transform = T.FixedPoints(args.pointnet_numpoints)
    else:
        transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])

    if args.use_all_set:
        dataset_retrieval = Kitti360CoarseDatasetMulti(
            args.base_path, SCENE_NAMES, transform, shuffle_hints=False, flip_poses=False,
            hint_type=args.hint_type, text_mode=args.text_mode, no_color = args.no_color
        )
    elif args.use_test_set:
        dataset_retrieval = Kitti360CoarseDatasetMulti(
            args.base_path, SCENE_NAMES_TEST, transform, shuffle_hints=False, flip_poses=False,
            hint_type=args.hint_type, text_mode=args.text_mode, no_color = args.no_color,
        )
    else:
        dataset_retrieval = Kitti360CoarseDatasetMulti(
            args.base_path, SCENE_NAMES_VAL, transform, shuffle_hints=False, flip_poses=False,
            hint_type=args.hint_type, text_mode=args.text_mode
        )
    
    dataloader_retrieval = DataLoader(
        dataset_retrieval,
        batch_size=args.batch_size,
        collate_fn=Kitti360CoarseDataset.collate_fn,
        shuffle=False,
    )

    # Load models
    model_coarse_dic = torch.load(args.path_coarse, map_location=torch.device("cpu"))
    model_coarse = CellRetrievalNetwork(
                KNOWN_CLASS,
                COLOR_NAMES_K360,
                args,
                is_eval = True, 
            )
    
    model_coarse.load_state_dict(model_coarse_dic, strict = False)
    model_coarse.to(device)
    
    # Run coarse
    retrievals, coarse_accuracies = run_coarse(model_coarse, dataloader_retrieval, args)
    print_accuracies(coarse_accuracies, "Coarse")

