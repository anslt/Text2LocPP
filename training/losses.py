from typing import List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict
import math

from datapreparation.kitti360pose.imports import Object3d, Pose, Cell
from scipy.optimize import linear_sum_assignment

from models.cross_matcher import get_pos_in_cell, get_pos_in_cell_intersect

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_directions, num_colors, num_classes, matcher, weight_dict, eos_coef):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_colors = num_colors
        self.num_directions = num_directions
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        
        empty_weight_class = torch.ones(self.num_classes + 1)
        empty_weight_class[-1] = self.eos_coef
        self.register_buffer('empty_weight_class', empty_weight_class)
        
        empty_weight_direction = torch.ones(self.num_directions + 1)
        empty_weight_direction[-1] = self.eos_coef
        self.register_buffer('empty_weight_direction', empty_weight_direction)

        empty_weight_color = torch.ones(self.num_colors + 1)
        empty_weight_color[-1] = self.eos_coef
        self.register_buffer('empty_weight_color', empty_weight_color)

    def loss_class(self, outputs, targets, indices, idx):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # assert 'pred_logits' in outputs
        src_logits = outputs['pred_class']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["class"][J] for t, (_, J) in zip(targets, indices)]).to(src_logits.device)
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        
        loss = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight_class)

        return loss
    
    def loss_direction(self, outputs, targets, indices, idx):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # assert 'pred_logits' in outputs
        src_logits = outputs['pred_direction']

        target_classes_o = torch.cat([t["direction"][J] for t, (_, J) in zip(targets, indices)]).to(src_logits.device)
        target_classes = torch.full(src_logits.shape[:2], self.num_directions,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight_direction)

        return loss
    
    def loss_color(self, outputs, targets, indices, idx):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        # assert 'pred_logits' in outputs
        src_logits = outputs['pred_color']

        target_classes_o = torch.cat([t["color"][J] for t, (_, J) in zip(targets, indices)]).to(src_logits.device)
        target_classes = torch.full(src_logits.shape[:2], self.num_colors,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight_color)

        return loss
    
    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        idx = self._get_src_permutation_idx(indices)
        loss_class = self.loss_class(outputs, targets, indices, idx)
        loss_color = self.loss_color(outputs, targets, indices, idx)
        loss_direction = self.loss_direction(outputs, targets, indices, idx)

        cost_class = self.weight_dict["class"]
        cost_color = self.weight_dict["color"]
        cost_direction = self.weight_dict["direction"]

        loss = cost_class * loss_class + cost_color * loss_color + cost_direction * loss_direction

        # print("class:" + str(round(loss_class.item(), 5)))
        # print("color:" + str(round(loss_color.item(), 5)))
        # print("direction:" + str(round(loss_direction.item(), 5)))

        return loss

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_direction: float = 1, cost_color: float = 1, cost_class: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_color = cost_color
        self.cost_direction = cost_direction
        assert cost_class != 0 or cost_color != 0 or cost_direction != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_class"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_class = outputs["pred_class"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_color = outputs["pred_color"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, 4]
        out_direction = outputs["pred_direction"].flatten(0, 1).softmax(-1) 

        # Also concat the target labels and boxes
        tgt_class = torch.cat([v["class"].to(out_class.device) for v in targets])
        tgt_color = torch.cat([v["color"].to(out_color.device)  for v in targets])
        tgt_direction = torch.cat([v["direction"].to(out_direction.device)  for v in targets])

        cost_class = -out_class[:, tgt_class]
        cost_color = -out_color[:, tgt_color]
        cost_direction = -out_direction[:, tgt_direction]

        # Final cost matrix
        C = self.cost_class * cost_class + self.cost_color * cost_color + self.cost_direction * cost_direction
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["class"]) for v in targets]
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]

class MatchingLoss(nn.Module):
    def __init__(self):
        """Matching loss for SuperGlue-based matching training"""
        super(MatchingLoss, self).__init__()
        self.eps = 1e-3

    # Matches as list[tensor ∈ [M1, 2], tensor ∈ [M2, 2], ...] for Mi: i ∈ [1, batch_size]
    def forward(self, P, all_matches):
        assert len(P.shape) == 3
        assert len(all_matches[0].shape) == 2 and all_matches[0].shape[-1] == 2
        assert len(P) == len(all_matches)
        batch_losses = []
        for i in range(len(all_matches)):
            matches = all_matches[i]
            cell_losses = -torch.log(P[i, matches[:, 0], matches[:, 1]])
            batch_losses.append(torch.mean(cell_losses))

        return torch.mean(torch.stack(batch_losses))


def calc_recall_precision(batch_gt_matches, batch_matches0, batch_matches1):
    assert len(batch_gt_matches) == len(batch_matches0) == len(batch_matches1)
    all_recalls = []
    all_precisions = []

    for idx in range(len(batch_gt_matches)):
        gt_matches, matches0, matches1 = (
            batch_gt_matches[idx],
            batch_matches0[idx],
            batch_matches1[idx],
        )
        gt_matches = gt_matches.tolist()

        recall = []
        for i, j in gt_matches:
            recall.append(matches0[i] == j or matches1[j] == i)

        precision = []
        for i, j in enumerate(matches0):
            if j >= 0:
                precision.append(
                    [i, j] in gt_matches
                )  # CARE: this only works as expected after tolist()

        recall = np.mean(recall) if len(recall) > 0 else 0.0
        precision = np.mean(precision) if len(precision) > 0 else 0.0
        all_recalls.append(recall)
        all_precisions.append(precision)

    return np.mean(all_recalls), np.mean(all_precisions)


def calc_pose_error_intersect(objects, matches0, poses: List[Pose], directions):
    assert len(objects) == len(matches0) == len(poses)
    assert isinstance(poses[0], Pose)

    batch_size, pad_size = matches0.shape
    poses = np.array([pose.pose for pose in poses])[:, 0:2]  # Assuming this is the best cell!

    errors = []
    for i_sample in range(batch_size):
        pose_prediction = get_pos_in_cell_intersect(
            objects[i_sample], matches0[i_sample], directions[i_sample]
        )
        errors.append(np.linalg.norm(poses[i_sample] - pose_prediction))
    return np.mean(errors)


def calc_pose_error(
    objects, matches0, poses: List[Pose], offsets=None, use_mid_pred=False, return_samples=False
):
    """Calculates the mean error of a batch by averaging the positions of all matches objects plus corresp. offsets.
    All calculations are in x-y-plane.

    Args:
        objects (List[List[Object3D]]): Objects-list for each sample in the batch.
        matches0 (np.ndarray): SuperGlue matching output of the batch.
        poses (np.ndarray): Ground-truth poses [batch_size, 3]
        offsets (List[np.ndarray], optional): List of offset vectors for all hints. Zero offsets are used if not given.
        use_mid_pred (bool, optional): If set, predicts the center of the cell regardless of matches and offsets. Defaults to False.

    Returns:
        [float]: Mean error.
    """
    assert len(objects) == len(matches0) == len(poses)
    assert isinstance(poses[0], Pose)

    batch_size, pad_size = matches0.shape
    poses = np.array([pose.pose for pose in poses])[:, 0:2]  # Assuming this is the best cell!

    if offsets is not None:
        assert len(objects) == len(offsets)
    else:
        offsets = np.zeros(
            (batch_size, pad_size, 2)
        )  # Set zero offsets to just predict the mean of matched-objects' centers

    errors = []
    for i_sample in range(batch_size):
        if use_mid_pred:
            pose_prediction = np.array((0.5, 0.5))
        else:
            pose_prediction = get_pos_in_cell(
                objects[i_sample], matches0[i_sample], offsets[i_sample]
            )
        errors.append(np.linalg.norm(poses[i_sample] - pose_prediction))

    if return_samples:
        return errors
    else:
        return np.mean(errors)
    
def calc_pose_error2(
    objects, poses: List[Pose], offsets=None, return_samples=False
):
    """Calculates the mean error of a batch by averaging the positions of all matches objects plus corresp. offsets.
    All calculations are in x-y-plane.

    Args:
        objects (List[List[Object3D]]): Objects-list for each sample in the batch.
        matches0 (np.ndarray): SuperGlue matching output of the batch.
        poses (np.ndarray): Ground-truth poses [batch_size, 3]
        offsets (List[np.ndarray], optional): List of offset vectors for all hints. Zero offsets are used if not given.
        use_mid_pred (bool, optional): If set, predicts the center of the cell regardless of matches and offsets. Defaults to False.

    Returns:
        [float]: Mean error.
    """
    assert len(objects) == len(poses)
    # assert isinstance(poses[0], Pose)

    # batch_size, pad_size = matches0.shape
    poses = np.array([pose.pose for pose in poses])[:, 0:2]  # Assuming this is the best cell!

    if offsets is not None:
        assert len(objects) == len(offsets)
    else:
        raise TypeError
        offsets = np.zeros(
            (batch_size, pad_size, 2)
        )  # Set zero offsets to just predict the mean of matched-objects' centers

    errors = []
    batch_size = len(poses)
    for i_sample in range(batch_size):
        # if use_mid_pred:
        #     pose_prediction = np.array((0.5, 0.5))
        # else:
        #     pose_prediction = get_pos_in_cell(
        #         objects[i_sample], matches0[i_sample], offsets[i_sample]
        #     )
        # if not is_degree:
        errors.append(np.linalg.norm(poses[i_sample] - offsets[i_sample]))
        # else:
        #     length, theta = offsets[i_sample]
        #     y = length * math.sin(theta) + 0.5
        #     x = length * math.cos(theta)
        #     errors.append(np.linalg.norm(poses[i_sample] - np.array([x, y])))

    if return_samples:
        return errors
    else:
        return np.mean(errors)

class CosineLoss(torch.nn.Module):
    def __init__(self,):
        """Pairwise Ranking loss for retrieval training.
        Implementation taken from a public GitHub, original paper:
        "Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models"
        (Kiros, Salakhutdinov, Zemel. 2014)

        Args:
            margin (float, optional): _description_. Defaults to 1.0.
        """
        super(CosineLoss, self).__init__()

    def forward(self, im, s):  # Norming the input (as in paper) is actually not helpful
        im = im / torch.norm(im, dim=1, keepdim=True)
        s = s / torch.norm(s, dim=1, keepdim=True)

        scores = torch.mm(im, s.transpose(1, 0))

        return (1 - scores).mean()

class PairwiseRankingLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        """Pairwise Ranking loss for retrieval training.
        Implementation taken from a public GitHub, original paper:
        "Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models"
        (Kiros, Salakhutdinov, Zemel. 2014)

        Args:
            margin (float, optional): _description_. Defaults to 1.0.
        """
        super(PairwiseRankingLoss, self).__init__()
        self.margin = margin

    def forward(self, im, s):  # Norming the input (as in paper) is actually not helpful
        im = im / torch.norm(im, dim=1, keepdim=True)
        s = s / torch.norm(s, dim=1, keepdim=True)

        margin = self.margin
        # compute image-sentence score matrix
        scores = torch.mm(im, s.transpose(1, 0))
        # print(scores)
        diagonal = scores.diag()

        # compare every diagonal score to scores in its column (i.e, all contrastive images for each sentence)
        cost_s = torch.max(
            torch.autograd.Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()),
            (margin - diagonal).expand_as(scores) + scores,
        )
        # compare every diagonal score to scores in its row (i.e, all contrastive sentences for each image)
        cost_im = torch.max(
            torch.autograd.Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()),
            (margin - diagonal).expand_as(scores).transpose(1, 0) + scores,
        )

        for i in range(scores.size()[0]):
            cost_s[i, i] = 0
            cost_im[i, i] = 0

        return (cost_s.sum() + cost_im.sum()) / len(im)  # Take mean for batch-size stability


class HardestRankingLoss(torch.nn.Module):
    def __init__(self, margin=1.0):
        super(HardestRankingLoss, self).__init__()
        self.margin = margin
        self.relu = nn.ReLU()

    def forward(self, images, captions):
        assert images.shape == captions.shape and len(images.shape) == 2
        images = images / torch.norm(images, dim=1, keepdim=True)
        captions = captions / torch.norm(captions, dim=1, keepdim=True)
        num_samples = len(images)

        similarity_scores = torch.mm(images, captions.transpose(1, 0))  # [I x C]

        cost_images = (
            self.margin + similarity_scores - similarity_scores.diag().view((num_samples, 1))
        )
        cost_images.fill_diagonal_(0)
        cost_images = self.relu(cost_images)
        cost_images, _ = torch.max(cost_images, dim=1)
        cost_images = torch.mean(cost_images)

        cost_captions = (
            self.margin
            + similarity_scores.transpose(1, 0)
            - similarity_scores.diag().view((num_samples, 1))
        )
        cost_captions.fill_diagonal_(0)
        cost_captions = self.relu(cost_captions)
        cost_captions, _ = torch.max(cost_captions, dim=1)
        cost_captions = torch.mean(cost_captions)

        cost = cost_images + cost_captions
        return cost

class ContrastiveLoss(torch.nn.Module):
    def __init__(self, temperature=1.0):
        """Pairwise Ranking loss for retrieval training.
        Implementation taken from a public GitHub, original paper:
        "Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models"
        (Kiros, Salakhutdinov, Zemel. 2014)

        Args:
            margin (float, optional): _description_. Defaults to 1.0.
        """
        super(ContrastiveLoss, self).__init__()
        # self.margin = margin
        self.temperature = temperature

    def forward(self, text_enc, obj_enc, mode = 0):  # Norming the input (as in paper) is actually not helpful

        text_enc = text_enc / torch.norm(text_enc, dim=1, keepdim=True)
        obj_enc = obj_enc / torch.norm(obj_enc, dim=1, keepdim=True)

        if mode == 0:

            similarity = torch.mm(text_enc, obj_enc.transpose(1, 0).contiguous())
            positives = torch.diag(similarity)

            numerator = torch.exp(positives / self.temperature)
            denominator = torch.exp(similarity / self.temperature) 
            
            all_losses = - torch.log(numerator / (torch.sum(denominator, dim=0))) - torch.log(numerator / (torch.sum(denominator, dim=1))) 
            loss = torch.mean(all_losses)

        elif mode == 1:
            
            batch_size = text_enc.shape[0]

            text_final = torch.cat([text_enc, obj_enc])
            obj_final = torch.cat([obj_enc, text_enc])

            similarity = torch.mm(text_final, obj_final.transpose(1, 0).contiguous())
            positives = torch.diag(similarity)

            numerator = torch.exp(positives / self.temperature)
            denominator = torch.exp(similarity / self.temperature) 

            mask = torch.ones_like(denominator)

            line1 = torch.arange(0, batch_size)
            line2 = torch.arange(batch_size, 2 * batch_size)
            
            mask[line1, line2] = 0.0
            mask[line2, line1] = 0.0

            denominator = denominator * mask

            all_losses = - torch.log(numerator / (torch.sum(denominator, dim=0))) - torch.log(numerator / (torch.sum(denominator, dim=1))) 
            loss = torch.mean(all_losses)

        elif mode == 2:

            similarity = torch.mm(text_enc, obj_enc.transpose(1, 0).contiguous())
            positives = torch.diag(similarity)

            numerator = torch.exp(positives / self.temperature)
            denominator = torch.exp(similarity / self.temperature) 
            
            all_losses = - torch.log(numerator / (torch.sum(denominator, dim=0))) - torch.log(numerator / (torch.sum(denominator, dim=1))) 
            loss = torch.mean(all_losses)


        else:
            raise TypeError 

        return loss

class CCL(nn.Module):
    """
    Compute contrastive loss
    """

    def __init__(self, temperature=0.05, method='log', q=0.5, ratio=0, alpha=2):
        super(CCL, self).__init__()
        self.tau = temperature
        self.method = method
        self.q = q
        self.ratio = ratio
        self.alpha = alpha
        self.thresh = 0.8

    def forward(self, im, s, mode = 0):

        eps = 1e-10
        im = im / (torch.norm(im, dim=1, keepdim=True) + eps)
        s = s / (torch.norm(s, dim=1, keepdim=True) + eps)
        scores = torch.mm(im, s.transpose(1, 0).contiguous())
        scores = (scores / self.tau).exp()
        i2t = scores / (scores.sum(1, keepdim=True) + eps)
        t2i = scores.t() / (scores.t().sum(1, keepdim=True) + eps)

        randn, eye = torch.rand_like(scores), torch.eye(scores.shape[0]).cuda()
        randn[eye > 0] = randn.min(dim=1)[0] - 1
        n = scores.shape[0]
        num = n - \
            1 if self.ratio <= 0 or self.ratio >= 1 else int(self.ratio * n)
        V, K = randn.topk(num, dim=1)
        mask = torch.zeros_like(scores)
        mask[torch.arange(n).reshape([-1, 1]).cuda(), K] = 1.

        # calculate feature distance
        dist = abs(torch.norm(im.unsqueeze(1)-s.unsqueeze(0), dim=2))
        dist = dist/dist.max()
        dist = pow((1-dist+1e-6), 1/self.alpha)
        dist[dist > 1] = 1
        dist[dist < self.thresh] = self.thresh

        if self.method == 'log':
            def criterion(x): return -((1. - x + eps).log())  # distance
        elif self.method == 'tan':
            def criterion(x): return (x.tan())
        elif self.method == 'abs':
            def criterion(x): return (x)
        elif self.method == 'exp':
            def criterion(x): return ((-(1. - x)).exp())
        elif self.method == 'gce':
            def criterion(x): return ((1. - (1. - x + eps) ** self.q) / self.q)
        else:
            raise Exception('Unknown Loss Function!')
        # return (criterion(i2t)*mask).sum(1).mean() + (criterion(t2i)*mask).sum(1).mean()
        out = (criterion(i2t) * dist*mask).sum(1).mean() + (criterion(t2i) * dist.permute(1, 0)*mask).sum(1).mean()
        # if torch.isnan(out) or torch.isinf(out):
        #     import pdb; pdb.set_trace()
        return out 

class ContrastiveLoss2(torch.nn.Module):
    def __init__(self, temperature=1.0):
        """Pairwise Ranking loss for retrieval training.
        Implementation taken from a public GitHub, original paper:
        "Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models"
        (Kiros, Salakhutdinov, Zemel. 2014)

        Args:
            margin (float, optional): _description_. Defaults to 1.0.
        """
        super(ContrastiveLoss2, self).__init__()
        # self.margin = margin
        self.temperature = temperature

    def forward(self, text_enc, obj_enc):  # Norming the input (as in paper) is actually not helpful

        text_enc = text_enc / torch.norm(text_enc, dim=1, keepdim=True)
        obj_enc = obj_enc / torch.norm(obj_enc, dim=1, keepdim=True)

        similarity = torch.mm(text_enc, obj_enc.transpose(1, 0).contiguous())
        similarity_text =  torch.mm(text_enc, text_enc.transpose(1, 0).contiguous())
        similarity_obj =  torch.mm(obj_enc, obj_enc.transpose(1, 0).contiguous())
        positives = torch.diag(similarity)

        numerator = torch.exp(positives / self.temperature)
        denominator = torch.exp(similarity / self.temperature)
        mask = 1 - torch.eye(len(similarity_text)).to(similarity_text.device)
        denominator_t =  torch.exp(similarity_text / self.temperature) * mask
        denominator_o =  torch.exp(similarity_obj / self.temperature) * mask
        
        all_losses = - torch.log(numerator / (torch.sum(denominator, dim=0) + torch.sum(denominator_t, dim=0))) \
                     - torch.log(numerator / (torch.sum(denominator, dim=1) + torch.sum(denominator_o, dim=0))) 
        loss = torch.mean(all_losses)  

        return loss


class ContrastiveLossWeight(torch.nn.Module):
    def __init__(self, temperature=1.0, max_weight=18.0):
        """Pairwise Ranking loss for retrieval training.
        Implementation taken from a public GitHub, original paper:
        "Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models"
        (Kiros, Salakhutdinov, Zemel. 2014)

        Args:
            margin (float, optional): _description_. Defaults to 1.0.
        """
        super(ContrastiveLossWeight, self).__init__()
        # self.margin = margin
        self.temperature = temperature
        self.max_weight = max_weight

    def forward(self, im, s, weight):  # Norming the input (as in paper) is actually not helpful

        im = im / torch.norm(im, dim=1, keepdim=True)
        s = s / torch.norm(s, dim=1, keepdim=True)

        similarity = torch.mm(im, s.transpose(1, 0).contiguous())
        positives = torch.diag(similarity)

        numerator = torch.exp(positives / self.temperature)
        denominator = torch.exp(similarity / self.temperature) 
        
        all_losses = - torch.log(numerator / (torch.sum(denominator, dim=0))) - torch.log(numerator / (torch.sum(denominator, dim=1))) 
        loss = torch.mean(all_losses)  

        return loss
    
class ContrastiveLossPos(torch.nn.Module):
    def __init__(self, temperature=1.0, threshold = 0.3333333333):
        """Pairwise Ranking loss for retrieval training.
        Implementation taken from a public GitHub, original paper:
        "Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models"
        (Kiros, Salakhutdinov, Zemel. 2014)

        Args:
            margin (float, optional): _description_. Defaults to 1.0.
        """
        super(ContrastiveLossPos, self).__init__()
        # self.margin = margin
        self.temperature = temperature
        self.threshold = threshold

    def forward(self, im, s, dist, index):  # Norming the input (as in paper) is actually not helpful

        im = im / torch.norm(im, dim=-1, keepdim=True) # N x C
        s = s / torch.norm(s, dim=-1, keepdim=True) # N x N_pos x C

        similarity = torch.bmm(im.unsqueeze(1), s.transpose(1, 2).contiguous()).squeeze(1) # N x N_pos
        positives = similarity[torch.arange(similarity.shape[0]).to(im.device), index]

        numerator = torch.exp(positives / self.temperature)
        denominator = torch.exp(similarity / self.temperature) * (dist > self.threshold)
        
        all_losses = - torch.log(numerator / (numerator + torch.sum(denominator, dim=1))) 
        loss = torch.mean(all_losses)  

        return loss
    
class ContrastiveLossCombine(torch.nn.Module):
    def __init__(self, temperature=1.0, threshold = 0.3333333333):
        """Pairwise Ranking loss for retrieval training.
        Implementation taken from a public GitHub, original paper:
        "Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models"
        (Kiros, Salakhutdinov, Zemel. 2014)

        Args:
            margin (float, optional): _description_. Defaults to 1.0.
        """
        super(ContrastiveLossCombine, self).__init__()
        # self.margin = margin
        self.temperature = temperature
        self.threshold = threshold

    def forward(self, im, s_org, dist, index):  # Norming the input (as in paper) is actually not helpful

        im = im / torch.norm(im, dim=-1, keepdim=True) # N x C
        s_org = s_org / torch.norm(s_org, dim=-1, keepdim=True) # N x N_pos x C
        s = s_org[torch.arange(s_org.shape[0]).to(s_org.device), index]

        similarity = torch.mm(im, s.transpose(1, 0).contiguous())
        positives = torch.diag(similarity)

        similarity2 = torch.bmm(im.unsqueeze(1), s_org.transpose(1, 2).contiguous()).squeeze(1) # N x N_pos
        # positives2 = similarity2[torch.arange(similarity2.shape[0]).to(im.device), index]

        numerator = torch.exp(positives / self.temperature)
        denominator = torch.exp(similarity / self.temperature) 

        denominator2 = torch.exp(similarity2 / self.temperature) * (dist > self.threshold)
        
        all_losses = - torch.log(numerator / (torch.sum(denominator, dim=0))) - torch.log(numerator / (torch.sum(denominator, dim=1)  + torch.sum(denominator2, dim=1) ))
        loss = torch.mean(all_losses) 

        return loss

class HardestRankingLoss(torch.nn.Module):
    def __init__(self, margin=1.0, scale = 64.0):
        super(HardestRankingLoss, self).__init__()
        self.margin = margin
        self.scale = scale
        self.relu = nn.ReLU()

    def forward(self, images, captions):
        # assert images.shape == captions.shape and len(images.shape) == 2
        # images = images / torch.norm(images, dim=1, keepdim=True)
        # captions = captions / torch.norm(captions, dim=1, keepdim=True)
        # num_samples = len(images)

        # similarity_scores = torch.mm(images, captions.transpose(1, 0))  # [I x C]

        # cost_images = (
        #     self.margin + similarity_scores - similarity_scores.diag().view((num_samples, 1))
        # )
        # cost_images.fill_diagonal_(0)
        # cost_images = self.relu(cost_images)
        # cost_images, _ = torch.max(cost_images, dim=1)
        # cost_images = torch.mean(cost_images)

        # cost_captions = (
        #     self.margin
        #     + similarity_scores.transpose(1, 0)
        #     - similarity_scores.diag().view((num_samples, 1))
        # )
        # cost_captions.fill_diagonal_(0)
        # cost_captions = self.relu(cost_captions)
        # cost_captions, _ = torch.max(cost_captions, dim=1)
        # cost_captions = torch.mean(cost_captions)

        # cost = cost_images + cost_captions
        # return cost * self.scale
        im = images / torch.norm(images, dim=1, keepdim=True)
        s = captions / torch.norm(captions, dim=1, keepdim=True)

        margin = self.margin
        # compute image-sentence score matrix
        scores = torch.mm(im, s.transpose(1, 0).contiguous())
        # print(scores)
        diagonal = scores.diag()

        # compare every diagonal score to scores in its column (i.e, all contrastive images for each sentence)
        cost_s = torch.max(
            torch.autograd.Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()),
            (margin - diagonal).expand_as(scores) + scores,
        )
        # print(cost_s)
        # compare every diagonal score to scores in its row (i.e, all contrastive sentences for each image)
        cost_im = torch.max(
            torch.autograd.Variable(torch.zeros(scores.size()[0], scores.size()[1]).cuda()),
            (margin - diagonal).expand_as(scores).transpose(1, 0) + scores,
        )

        for i in range(scores.size()[0]):
            cost_s[i, i] = 0
            cost_im[i, i] = 0
        
        # import pdb; pdb.set_trace()
        
        cost_captions, _ = torch.max(cost_s, dim=1)
        cost_captions = torch.mean(cost_captions)

        cost_images, _ = torch.max(cost_im, dim=1)
        cost_images = torch.mean(cost_images)

        cost = cost_images + cost_captions
        return cost * self.scale


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature, world_size):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.world_size = world_size

        self.mask = self.mask_correlated_samples(batch_size, world_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size, world_size):
        N = 2 * batch_size * world_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size * world_size):
            mask[i, batch_size * world_size + i] = 0
            mask[batch_size * world_size + i, i] = 0
        return mask

    def forward(self, z_i, z_j):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size * self.world_size

        z = torch.cat((z_i, z_j), dim=0)
        if self.world_size > 1:
            z = torch.cat(GatherLayer.apply(z), dim=0)

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size * self.world_size)
        sim_j_i = torch.diag(sim, -self.batch_size * self.world_size)

        # We have 2N samples, but with Distributed training every GPU gets N examples too, resulting in: 2xNxN
        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

# https://github.com/ae-foster/pytorch-simclr/blob/simclr-master/critic.py   
class LinearCritic(nn.Module):

    def __init__(self, temperature=1.):
        super(LinearCritic, self).__init__()
        self.temperature = temperature
        self.cossim = nn.CosineSimilarity(dim=-1)
        self.criterion = nn.CrossEntropyLoss()
    
    def forward(self, z1, z2):
    # def forward(self, h1, h2):
        # z1, z2 = self.project(h1), self.project(h2)
        sim11 = self.cossim(z1.unsqueeze(-2), z1.unsqueeze(-3)) / self.temperature
        sim22 = self.cossim(z2.unsqueeze(-2), z2.unsqueeze(-3)) / self.temperature
        sim12 = self.cossim(z1.unsqueeze(-2), z2.unsqueeze(-3)) / self.temperature
        d = sim12.shape[-1]
        sim11[..., range(d), range(d)] = float('-inf')
        sim22[..., range(d), range(d)] = float('-inf')
        raw_scores1 = torch.cat([sim12, sim11], dim=-1)
        raw_scores2 = torch.cat([sim22, sim12.transpose(-1, -2)], dim=-1)
        raw_scores = torch.cat([raw_scores1, raw_scores2], dim=-2)
        targets = torch.arange(2 * d, dtype=torch.long, device=raw_scores.device)
        return self.criterion(raw_scores, targets)

if __name__ == "__main__":
    objects = [
        [
            EasyDict(center=np.array([0, 0])),
            EasyDict(center=np.array([10, 10])),
            EasyDict(center=np.array([99, 99])),
        ],
    ]
    matches0 = np.array((0, 1, -1)).reshape((1, 3))
    poses = np.array((0, 10)).reshape((1, 2))
    offsets = np.array([(2, 10), (-10, 0)]).reshape((1, 2, 2))

    err = calc_pose_error(objects, matches0, poses, offsets=None)
    print(err)
    err = calc_pose_error(objects, matches0, poses, offsets=offsets)
    print(err)
