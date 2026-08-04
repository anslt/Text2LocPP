"""Module for training the coarse cell-retrieval module
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch_geometric.transforms as T
import collections

import time
import numpy as np
import matplotlib.pyplot as plt
import cv2
from easydict import EasyDict
import os
import os.path as osp
import tqdm
import json
from nltk import tokenize as text_tokenize

from models.cell_retrieval import CellRetrievalNetwork

# from configs.mixed0 import SCENE_NAMES, SCENE_NAMES_VAL, SCENE_NAMES_TEST, SCENE_NAMES_TRAIN
from configs.mixed0 import COLOR_NAMES as COLOR_NAMES_K360
from configs.mixed0 import CLASS_TO_INDEX, DIRECTION_TO_INDEX, COLOR_TO_INDEX
from dataloading.kitti360pose.cells import Kitti360CoarseDatasetMulti, Kitti360CoarseDataset

from training.args import parse_arguments
from training.plots import plot_metrics
from training.losses import MatchingLoss, PairwiseRankingLoss, HardestRankingLoss, ContrastiveLoss, ContrastiveLoss2  
from training.losses import ContrastiveLossPos, ContrastiveLossWeight, ContrastiveLossCombine
from training.losses import HungarianMatcher, SetCriterion, CosineLoss
from training.utils import plot_retrievals
from scipy.optimize import linear_sum_assignment

from nltk import tokenize as text_tokenize

def train_epoch(model, dataloader, args):
    model.train()
    epoch_losses = []

    batches = []
    pbar = tqdm.tqdm(enumerate(dataloader), total = len(dataloader))

    if args.replace_color:
        assert args.no_color

    for i_batch, batch in pbar:

        optimizer.zero_grad()
        assert batch["length"][0] != 0
        torch.cuda.empty_cache()

        if args.sub_sample:
            lang_sub = batch["sub_text"]
            lang_sub_gt = batch["sub_gt"]

            if args.aux_split_type == "sentence":
                length_sub = [len(text_tokenize.sent_tokenize(sub_text)) for sub_text in lang_sub]
            else:
                length_sub = [1] * len(lang_sub)
            length_sub_gt = [len(sub_text.split(". ")) for sub_text in lang_sub_gt]
            fake_hints_number = [1] * len(lang_sub) # no use just fit the sample
            anchor, logits_aux_dict, inst_t = model.encode_aux_text(lang_sub, length_sub, fake_hints_number, fold = args.fold)
            positive, logits_dict, inst_t = model.encode_text(lang_sub_gt, length_sub_gt, fake_hints_number)
        else:
            lang = batch["texts1"]
            anchor, logits_aux_dict, inst_t  = model.encode_aux_text(lang, batch["length"], batch["texts1_number"], fold = args.fold)
            positive, logits_dict, inst_t  = model.encode_text(batch["texts"])

        if args.ranking_loss == "triplet":
            negative_cell_objects = [cell.objects for cell in batch["negative_cells"]]
            negative = model.encode_objects(negative_cell_objects)
            loss = criterion(anchor, positive, negative)
        # elif args.weight_loss:
        #     loss = criterion(anchor, positive, weight_result)
        # elif args.combine_anchor_loss:
        #     loss = criterion(anchor, positive_org, dist, index)
        else:
            loss = criterion(anchor, positive) # anchor (texts) , positive (objects)

        if args.org_text:
            loss += args.org_text_coeff * criterion(anchor, anchor)

        if args.aux_intra_loss:
            targets = []
            for batch_target in batch["sub_target"]:
                for target in batch_target:
                    targets.append(target)
            loss += aux_creterion(logits_aux_dict, targets)


        loss.backward()
        optimizer.step()
        # print(torch.cuda.memory_summary(device=None, abbreviated=False))

        epoch_losses.append(loss.item())
        # print(loss.item())
        # dataloader.dataset.update_limit(6)
        # import pdb; pdb.set_trace()
        # for para in model.language_encoder.parameters():
        #     print(para)
        #     para.require_grads = False

    return np.mean(epoch_losses), batches


@torch.no_grad()
def eval_epoch(model, dataloader, args, return_encodings=False, return_distance=False, aux = True, origin = False):
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


if __name__ == "__main__":
    args = parse_arguments()
    print(str(args).replace(",", "\n"), "\n")

    dataset_name = args.base_path[:-1] if args.base_path.endswith("/") else args.base_path
    dataset_name = dataset_name.split("/")[-1]
    print(f"Directory: {dataset_name}")

    config_file = json.load(open(os.path.join("configs", args.configs + ".json")))
    SCENE_NAMES = config_file["SCENE_NAMES"]
    SCENE_NAMES_TRAIN = config_file["SCENE_NAMES_TRAIN"]
    SCENE_NAMES_VAL = config_file["SCENE_NAMES_VAL"]
    SCENE_NAMES_TEST = config_file["SCENE_NAMES_TEST"]

    cont = "Y" if bool(args.continue_path) else "N"
    feats = "all" if len(args.use_features) == 3 else "-".join(args.use_features)
    folder_name = args.folder_name
    print("#####################")
    print("########   Folder Name: " + folder_name)
    print("#####################")
    if not osp.isdir(f"./checkpoints/{dataset_name}"):
        os.mkdir(f"./checkpoints/{dataset_name}")
    if not osp.isdir(f"./checkpoints/{dataset_name}/{folder_name}"):
        os.mkdir(f"./checkpoints/{dataset_name}/{folder_name}")

    """
    Create data loaders
    """
    if args.dataset == "K360":
        # ['2013_05_28_drive_0003_sync', ]
        if args.no_pc_augment:
            train_transform = T.FixedPoints(args.pointnet_numpoints)
            val_transform = T.FixedPoints(args.pointnet_numpoints)
        else:
            train_transform = T.Compose(
                [
                    T.FixedPoints(args.pointnet_numpoints),
                    T.RandomRotate(120, axis=2),
                    T.NormalizeScale(),
                ]
            )
            val_transform = T.Compose([T.FixedPoints(args.pointnet_numpoints), T.NormalizeScale()])

        dataset_train = Kitti360CoarseDatasetMulti(
            args.base_path,
            SCENE_NAMES_TRAIN,
            train_transform,
            shuffle_hints=True,
            flip_poses=True,
            hint_type=args.hint_type,
            add_object_number=args.add_object_number,
            random_choice_direction=args.random_choice_direction,
            replace_pose=args.replace_pose,
            no_color=args.no_color0,
            replace_color=args.replace_color,
            is_global=args.is_global,
            constraint_all=args.constraint_all,
            random_discard=args.random_discard,
            text_mode=args.text_mode,
            sub_sampie=args.sub_sample,
            transfer_mode=args.transfer_mode,
            output_detail=args.aux_intra_loss, # is you use intra loss, you need output detail
            split_type=args.aux_split_type
        )

        if args.debug:
            import pdb; pdb.set_trace()

        if args.aux_prepare:
            dataset_train.update_limit(args.aux_limit)

        dataloader_train = DataLoader(
            dataset_train,
            batch_size=args.batch_size,
            collate_fn=Kitti360CoarseDataset.collate_fn,
            shuffle=args.shuffle,
            num_workers=args.cpus,
            drop_last=True,
        )

        dataset_val = Kitti360CoarseDatasetMulti(args.base_path, SCENE_NAMES_VAL, val_transform,hint_type=args.hint_type,
                                                 add_object_number=args.add_object_number,no_color=args.no_color0, replace_color=args.replace_color,
                                                 is_global=args.is_global, text_mode=args.text_mode,)

        dataloader_val = DataLoader(
            dataset_val,
            batch_size=1,
            collate_fn=Kitti360CoarseDataset.collate_fn,
            shuffle=False,
        )

        if len(SCENE_NAMES_TEST) > 0:

            dataset_test = Kitti360CoarseDatasetMulti(args.base_path, SCENE_NAMES_TEST, val_transform,hint_type=args.hint_type,
                                                    add_object_number=args.add_object_number,no_color=args.no_color0, replace_color=args.replace_color,
                                                    is_global=args.is_global,text_mode=args.text_mode,)

            dataloader_test = DataLoader(
                dataset_test,
                batch_size=1,
                collate_fn=Kitti360CoarseDataset.collate_fn,
                shuffle=False,
            )

    assert sorted(dataset_train.get_known_classes()) == sorted(dataset_val.get_known_classes())

    data = dataset_train[0]
    # assert len(data["debug_hint_descriptions"]) == args.num_mentioned
    # maybe add something
    batch = next(iter(dataloader_train))

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("device:", device, torch.cuda.get_device_name(0))
    torch.autograd.set_detect_anomaly(True)

    lr = args.learning_rate

    dict_loss = {lr: []}
    dict_acc = {k: [] for k in args.top_k}
    dict_acc_val = {k: [] for k in args.top_k}
    dict_acc_val_close = {k: [] for k in args.top_k}
    if len(SCENE_NAMES_TEST) > 0:
        dict_acc_test = {k: [] for k in args.top_k}
        dict_acc_test_close = {k: [] for k in args.top_k}

    best_val_accuracy = -1
    last_model_save_path_val = None

    model = CellRetrievalNetwork(
            dataset_train.get_known_classes(),
            COLOR_NAMES_K360,
            args,
            is_eval = args.finetune # no need load pretrained pointnet++
        )
    if args.continue_path:
        model_dic = torch.load(args.continue_path, map_location=torch.device("cpu"))
        model.load_state_dict(model_dic, strict = False)
    elif args.finetune_path:
        model_dic = torch.load(args.finetune_path, map_location=torch.device("cpu"))
        try:
            model.load_state_dict(model_dic)
        except Exception as e:
            # print(e)
            print("Not strict loading!")
            model.load_state_dict(model_dic, strict = False)
            

    model.to(device)

    
    if args.weight_loss:
        assert args.ranking_loss == "contrastive"
        ele_num = 2 if args.no_color else 3
        criterion = ContrastiveLossWeight(temperature=args.temperature, max_weight = ele_num * args.num_mentioned)
    else:
        if args.ranking_loss == "pairwise":
            criterion = PairwiseRankingLoss(margin=args.margin)
        if args.ranking_loss == "hardest":
            criterion = HardestRankingLoss(margin=args.margin)
        if args.ranking_loss == "triplet":
            criterion = nn.TripletMarginLoss(margin=args.margin)
        if args.ranking_loss == "contrastive":            
            if args.combine_anchor_loss:
                criterion = ContrastiveLossCombine(temperature=args.temperature, threshold=args.pos_threshold)
            else:
                criterion = ContrastiveLoss(temperature=args.temperature)
        if args.ranking_loss == "contrastive2":
            criterion = ContrastiveLoss2(temperature=args.temperature)
        if args.ranking_loss == "cosine":
            criterion = CosineLoss()  

    if args.text_loss:
        if args.temperature_text != 0:
            temperature_text = args.temperature_text
        else:
            temperature_text = args.temperature

        criterion_text = ContrastiveLoss(temperature=temperature_text)
    
    if args.constraint_all:
        criterion2 = nn.CrossEntropyLoss()

    if args.add_anchor and not args.combine_anchor_loss:
        criterion_pos = ContrastiveLossPos(temperature=args.temperature, threshold=args.pos_threshold)

    if args.aux_intra_loss:
        matcher = HungarianMatcher(args.intra_cost_direction, args.intra_cost_color,  args.intra_cost_class)
        # matcher.to(device)
        weight_dict = {"direction": args.intra_cost_direction, "color": args.intra_cost_color, "class": args.intra_cost_class}
        eos_coef = args.intra_eos_coef
        aux_creterion = SetCriterion(len(DIRECTION_TO_INDEX), len(COLOR_TO_INDEX), len(CLASS_TO_INDEX) - 1, matcher, weight_dict, eos_coef)
        aux_creterion.to(device)
        
    # if args.finetune:
    #     val_acc, val_acc_close, val_retrievals = eval_epoch(model, dataloader_val, args, aux = False, origin = True)
        
    #     if len(SCENE_NAMES_TEST) > 0:
    #         test_acc, test_acc_close, test_retrievals = eval_epoch(model, dataloader_test, args)

    #     print(f"\t Before Training, ", end="")
    #     # for k, v in train_acc.items():
    #     #     print(f"{k}-{v:0.3f} ", end="")
    #     print("val-acc: ", end="")

    #     for k, v in val_acc.items():
    #         print(f"{k}-{v:0.3f} ", end="")
    #     print("val-acc-close: ", end="")
    #     for k, v in val_acc_close.items():
    #         print(f"{k}-{v:0.3f} ", end="")

    #     if len(SCENE_NAMES_TEST) > 0:
    #         print("test-acc: ", end="")
    #         for k, v in test_acc.items():
    #             print(f"{k}-{v:0.3f} ", end="")
    #         print("test-acc-close: ", end="")
    #         for k, v in test_acc_close.items():
    #             print(f"{k}-{v:0.3f} ", end="")
    #     print("\n", flush=True)

    if args.transfer_mode:
        temporary = model.language_encoder
        model.language_encoder = model.aux_language_encoder
        model.aux_language_encoder = temporary

    for para in model.language_encoder.parameters():
        para.requires_grad = False
    for para in model.object_encoder.parameters():
        para.requires_grad = False
    for para in model.obj_inter_module.parameters():
        para.requires_grad = False

    model_llm_model = None
    if args.memory_efficient:
        assert torch.cuda.is_available()
        multi_flag = False
        if torch.cuda.device_count() > 1:
            multi_flag = True
            model.aux_language_encoder.llm_model = model.aux_language_encoder.llm_model.cuda(1)
            # model.aux_language_encoder.llm_model = None
        else:
            temporary = model.object_encoder
            model.object_encoder = None
            temporary.to("cpu")

    optimizer = optim.Adam(model.parameters(), lr=lr)

    if args.lr_scheduler == "exponential":
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, args.lr_gamma)
    elif args.lr_scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, args.lr_step, args.lr_gamma)
    else:
        raise TypeError

    for epoch in range(1, args.epochs + 1):
        # dataset_train.reset_seed() #OPTION: re-setting seed leads to equal data at every epoch
        if args.aux_prepare:
            if epoch == args.aux_epochs + 1:
                dataloader_train.dataset.update_limit(6)
            elif epoch == args.aux_epochs2 + 1:
                dataloader_train.dataset.update_limit(9)
            print(dataloader_train.dataset.aux_limit)
        
        loss, train_batches = train_epoch(model, dataloader_train, args)
        # loss = 0

        if args.memory_efficient:

            if multi_flag:
                model.aux_language_encoder.llm_model = model.aux_language_encoder.llm_model.cuda(0)
            else:

                temporary.to("cuda")
                model.object_encoder = temporary

        # train_acc, train_acc_close, train_retrievals = eval_epoch(
        #     model, dataloader_train, args
        # )  
        # train_acc, train_acc_close, train_retrievals = 0.0, 0.0, None
        val_acc, val_acc_close, val_retrievals = eval_epoch(model, dataloader_val, args)
        if len(SCENE_NAMES_TEST) > 0:
            test_acc, test_acc_close, test_retrievals = eval_epoch(model, dataloader_test, args)
        else:
            test_acc = None

        if args.memory_efficient:
            
            if multi_flag:
                
                model.aux_language_encoder.llm_model = model.aux_language_encoder.llm_model.cuda(1)

            else:
                model.object_encoder = None
                temporary.to("cpu")

        key = lr
        dict_loss[key].append(loss)
        for k in args.top_k:
            # dict_acc[k].append(train_acc[k])
            dict_acc_val[k].append(val_acc[k])
            dict_acc_val_close[k].append(val_acc_close[k])
            if len(SCENE_NAMES_TEST) > 0:
                dict_acc_test[k].append(test_acc[k])
                dict_acc_test_close[k].append(test_acc_close[k])

        scheduler.step()
        print(f"\t lr {lr:0.4} loss {loss:0.3f} epoch {epoch} ", end="")
        # for k, v in train_acc.items():
        #     print(f"{k}-{v:0.3f} ", end="")
        print("val-acc: ", end="")
        for k, v in val_acc.items():
            print(f"{k}-{v:0.3f} ", end="")
        print("val-acc-close: ", end="")
        for k, v in val_acc_close.items():
            print(f"{k}-{v:0.3f} ", end="")

        if len(SCENE_NAMES_TEST) > 0:
            print("test-acc: ", end="")
            for k, v in test_acc.items():
                print(f"{k}-{v:0.3f} ", end="")
            print("test-acc-close: ", end="")
            for k, v in test_acc_close.items():
                print(f"{k}-{v:0.3f} ", end="")
        print("\n", flush=True)

        # Saving best model
        if test_acc is not None:
            acc = test_acc
        else:
            acc = val_acc
        acc_val = sum(acc.values())# val_acc[max(args.top_k)]
        max_val = acc[max(args.top_k)]
        if acc_val > best_val_accuracy:
            model_path = f"./checkpoints/{dataset_name}/{folder_name}/coarse_cont{cont}_epoch{epoch}_acc{max_val:0.3f}_ecl{int(args.class_embed)}_eco{int(args.color_embed)}_p{args.pointnet_numpoints}_npa{int(args.no_pc_augment)}_loss-{args.ranking_loss}_f-{feats}.pth"
            if not osp.isdir(osp.dirname(model_path)):
                os.mkdir(osp.dirname(model_path))

            print(f"Saving model at {acc_val:0.2f} to {model_path}")
            
            try:

                model_dic = model.state_dict()
                out = collections.OrderedDict()
                for item in model_dic:
                    if "llm_model" not in item:
                        out[item] = model_dic[item]
                torch.save(out, model_path)
                if (
                    last_model_save_path_val is not None
                    and last_model_save_path_val != model_path
                    and osp.isfile(last_model_save_path_val)
                ):  
                    print("Removing", last_model_save_path_val)
                    os.remove(last_model_save_path_val)
                
                last_model_save_path_val = model_path
                
            except Exception as e:
                print(f"Error saving model!", str(e))
            best_val_accuracy = acc_val
                           

