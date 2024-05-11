import torch
import torch.nn.functional as F
from torch import nn
import numpy as np

from .utils import Matcher, BalancedPositiveNegativeSampler, roi_align, get_initial_points
from .box_ops import BoxCoder, box_iou, process_box, nms


def dice_loss(input, target):
    'input must be the result of sigmoid'
    smooth = 1.

    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    
    return 1 - ((2. * intersection + smooth) /
              (iflat.sum() + tflat.sum() + smooth))


def fastrcnn_loss(class_logit, box_regression, label, regression_target):
    classifier_loss = F.cross_entropy(class_logit, label)

    N, num_pos = class_logit.shape[0], regression_target.shape[0]
    box_regression = box_regression.reshape(N, -1, 4)
    box_regression, label = box_regression[:num_pos], label[:num_pos]
    box_idx = torch.arange(num_pos, device=label.device)

    box_reg_loss = F.smooth_l1_loss(box_regression[box_idx, label], regression_target, reduction='sum') / N

    return classifier_loss, box_reg_loss


def maskrcnn_loss(mask_logit, proposal, matched_idx, label, gt_mask):
    matched_idx = matched_idx[:, None].to(proposal)
    roi = torch.cat((matched_idx, proposal), dim=1)

    M = mask_logit.shape[-1]
    gt_mask = gt_mask[:, None].to(roi)
    mask_target = roi_align(gt_mask, roi, 1., M, M, -1)[:, 0]

    idx = torch.arange(label.shape[0], device=label.device)
    mask_loss = F.binary_cross_entropy_with_logits(mask_logit[idx, label], mask_target)
    return mask_loss


def edgercnn_loss(edge_logit, proposal, matched_idx, label, gt_edge):
    matched_idx = matched_idx[:, None].to(proposal)
    roi = torch.cat((matched_idx, proposal), dim=1)

    M = edge_logit.shape[-1]
    gt_edge = gt_edge[:, None].to(roi)
    edge_target = roi_align(gt_edge, roi, 1., M, M, -1)[:, 0]

    idx = torch.arange(label.shape[0], device=label.device)
    edge_bce = F.binary_cross_entropy_with_logits(edge_logit[idx, 0], edge_target, pos_weight=torch.Tensor([28*28/50]).to(idx.device))
    edge_dice = dice_loss(torch.sigmoid(edge_logit[idx, 0]), edge_target)
    edge_loss = edge_bce + edge_dice
    return 0.2 * edge_loss


def vertexrcnn_loss(vertex_logit, proposal, matched_idx, label, gt_vertex):
    matched_idx = matched_idx[:, None].to(proposal)
    roi = torch.cat((matched_idx, proposal), dim=1)

    M = vertex_logit.shape[-1]
    gt_vertex = gt_vertex[:, None].to(roi)
    vertex_target = roi_align(gt_vertex, roi, 1., M, M, -1)[:, 0]

    idx = torch.arange(label.shape[0], device=label.device)
    vertex_loss = F.binary_cross_entropy_with_logits(vertex_logit[idx, 0], vertex_target, pos_weight=torch.Tensor([28*28/16*3]).to(idx.device))
    return vertex_loss


def poly_matching_loss(pnum, pred, gt, matched_idx, loss_type="L1"):

    matched_idx = matched_idx.unsqueeze(1).unsqueeze(2).long().to(gt.device)
    matched_idx = matched_idx.expand(-1, gt.shape[1], gt.shape[2])
    # print('--------------------------')
    # print(f'gt shape: {gt.shape}, matched_idx shape: {matched_idx2.shape}')
    # print(f'matched idx max: {matched_idx.max()}, matched idx min: {matched_idx2.min()}')
    # print(f'matched idx: {matched_idx}')
    # print('--------------------------')
    gt = torch.gather(gt, 0, matched_idx)

    batch_size = pred.size()[0]
    pidxall = np.zeros(shape=(batch_size, pnum, pnum), dtype=np.int32)

    for b in range(batch_size):
        for i in range(pnum):
            pidx = (np.arange(pnum) + i) % pnum
            pidxall[b, i] = pidx

    pidxall = torch.from_numpy(np.reshape(pidxall, newshape=(batch_size, -1))).to(pred.device)

    feature_id = pidxall.unsqueeze_(2).long().expand(pidxall.size(0), pidxall.size(1), gt.size(2)).detach()
    gt_expand = torch.gather(gt, 1, feature_id).view(batch_size, pnum, pnum, 2)

    pred_expand = pred.unsqueeze(1)

    dis = pred_expand - gt_expand

    if loss_type == "L2":
        dis = (dis ** 2).sum(3).sqrt().sum(2)
    elif loss_type == "L1":
        dis = torch.abs(dis).sum(3).sum(2)

    min_dis, min_id = torch.min(dis, dim=1, keepdim=True)
    min_id = torch.from_numpy(min_id.data.cpu().numpy()).to(pred.device)
    min_gt_id_to_gather = min_id.unsqueeze_(2).unsqueeze_(3).long().expand(min_id.size(0),
                                                                           min_id.size(1),
                                                                           gt_expand.size(2),
                                                                           gt_expand.size(3))

    gt_right_order = torch.gather(gt_expand, 1, min_gt_id_to_gather).view(batch_size, pnum, 2)  # TODO: check the application of this

    return gt_right_order, 0.25 * torch.mean(min_dis)


class RoIHeads(nn.Module):
    def __init__(self, box_roi_pool, box_predictor,
                 fg_iou_thresh, bg_iou_thresh,
                 num_samples, positive_fraction,
                 reg_weights,
                 score_thresh, nms_thresh, num_detections,
                 coarse_to_fine_steps=3, num_points=16):
        super().__init__()
        self.box_roi_pool = box_roi_pool
        self.box_predictor = box_predictor

        self.mask_roi_pool = None
        self.augmentation_roi_pool = None
        self.mask_predictor = None

        self.proposal_matcher = Matcher(fg_iou_thresh, bg_iou_thresh, allow_low_quality_matches=False)
        self.fg_bg_sampler = BalancedPositiveNegativeSampler(num_samples, positive_fraction)
        self.box_coder = BoxCoder(reg_weights)

        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        self.num_detections = num_detections
        self.min_size = 1

        self.coarse_to_fine_steps = coarse_to_fine_steps
        self.num_points = num_points

    def has_mask(self):
        if self.mask_roi_pool is None:
            return False
        if self.mask_predictor is None:
            return False
        return True

    def select_training_samples(self, proposal, target):
        gt_box = target['boxes']
        gt_label = target['labels']
        proposal = torch.cat((proposal, gt_box))

        iou = box_iou(gt_box, proposal)
        pos_neg_label, matched_idx = self.proposal_matcher(iou)
        pos_idx, neg_idx = self.fg_bg_sampler(pos_neg_label)
        idx = torch.cat((pos_idx, neg_idx))

        regression_target = self.box_coder.encode(gt_box[matched_idx[pos_idx]], proposal[pos_idx])
        proposal = proposal[idx]
        matched_idx = matched_idx[idx]
        label = gt_label[matched_idx]
        num_pos = pos_idx.shape[0]
        label[num_pos:] = 0

        return proposal, matched_idx, label, regression_target

    def fastrcnn_inference(self, class_logit, box_regression, proposal, image_shape):
        N, num_classes = class_logit.shape

        device = class_logit.device
        pred_score = F.softmax(class_logit, dim=-1)
        box_regression = box_regression.reshape(N, -1, 4)

        boxes = []
        labels = []
        scores = []
        for l in range(1, num_classes):
            score, box_delta = pred_score[:, l], box_regression[:, l]

            keep = score >= self.score_thresh
            box, score, box_delta = proposal[keep], score[keep], box_delta[keep]
            box = self.box_coder.decode(box_delta, box)

            box, score = process_box(box, score, image_shape, self.min_size)

            keep = nms(box, score, self.nms_thresh)[:self.num_detections]
            box, score = box[keep], score[keep]
            label = torch.full((len(keep),), l, dtype=keep.dtype, device=device)

            boxes.append(box)
            labels.append(label)
            scores.append(score)

        results = dict(boxes=torch.cat(boxes), labels=torch.cat(labels), scores=torch.cat(scores))
        return results

    def forward(self, feature, proposal, image_shape, target):
        if self.training:
            proposal, matched_idx, label, regression_target = self.select_training_samples(proposal, target)
        w, h = image_shape
        box_feature = self.box_roi_pool(feature, [proposal], [(w, h)])
        class_logit, box_regression = self.box_predictor(box_feature)

        result, losses = {}, {}
        if self.training:
            classifier_loss, box_reg_loss = fastrcnn_loss(class_logit, box_regression, label, regression_target)
            losses = dict(roi_classifier_loss=classifier_loss, roi_box_loss=box_reg_loss)
        else:
            result = self.fastrcnn_inference(class_logit, box_regression, proposal, image_shape)

        if self.has_mask():
            if self.training:
                num_pos = regression_target.shape[0]

                mask_proposal = proposal[:num_pos]
                pos_matched_idx = matched_idx[:num_pos]
                mask_label = label[:num_pos]

                '''
                # -------------- critial ----------------
                box_regression = box_regression[:num_pos].reshape(num_pos, -1, 4)
                idx = torch.arange(num_pos, device=mask_label.device)
                mask_proposal = self.box_coder.decode(box_regression[idx, mask_label], mask_proposal)
                # ---------------------------------------
                '''

                if mask_proposal.shape[0] == 0:
                    losses.update(dict(roi_mask_loss=torch.tensor(0), roi_polygon_loss=torch.tensor(0),
                                       roi_edge_loss=torch.tensor(0), roi_vertex_loss=torch.tensor(0)))
                    return result, losses
            else:
                mask_proposal = result['boxes']

                if mask_proposal.shape[0] == 0:
                    result.update(dict(masks=torch.empty((0, 28, 28)),
                                       edges=torch.empty((0, 28, 28)), vertices=torch.empty((0, 28, 28)),
                                       polygons=torch.empty((0, self.num_points, 2)),
                                       adjacency=torch.empty((0, self.num_points, self.num_points))))
                    return result, losses

            mask_feature = self.mask_roi_pool(feature, [mask_proposal], [(w, h)])
            mask_logit = self.mask_predictor(mask_feature)

            augmentation_feature = self.augmentation_roi_pool(feature, [mask_proposal], [(w, h)])
            edge_logit, vertex_logit = self.feature_augmentor(augmentation_feature)
            # Feature augmentation
            enhanced_feature = torch.cat([augmentation_feature, edge_logit, vertex_logit], 1)
            poly_feature = self.poly_augmentor(enhanced_feature)

            # GCN
            # create circle polygon data
            init_polys = get_initial_points(self.num_points)
            init_polys = torch.from_numpy(init_polys).unsqueeze(0).repeat(poly_feature.shape[0], 1, 1)
            polygcn_feature = poly_feature.permute(0, 2, 3, 1).view(-1, poly_feature.shape[-1]**2, poly_feature.shape[1])
            pred_polygon, pred_adjacent = self.polygon_predictor(polygcn_feature, init_polys)

            if self.training:
                gt_mask = target['masks']
                gt_edge = target['edges']
                gt_vertex = target['vertices']
                gt_polygon = target['polygons']

                mask_loss = maskrcnn_loss(mask_logit, mask_proposal, pos_matched_idx, mask_label, gt_mask)
                edge_loss = edgercnn_loss(edge_logit, mask_proposal, pos_matched_idx, mask_label, gt_edge)
                vertex_loss = vertexrcnn_loss(vertex_logit, mask_proposal, pos_matched_idx, mask_label, gt_vertex)
                _, polygon_loss = poly_matching_loss(self.num_points, pred_polygon, gt_polygon, pos_matched_idx, loss_type="L1")
                losses.update(dict(roi_mask_loss=mask_loss, roi_edge_loss=edge_loss,
                                   roi_vertex_loss=vertex_loss, roi_polygon_loss=polygon_loss))
            else:
                label = result['labels']
                idx = torch.arange(label.shape[0], device=label.device)

                mask_logit = mask_logit[idx, label]
                edge_logit = edge_logit[idx, 0]
                vertex_logit = vertex_logit[idx, 0]

                mask_prob = mask_logit.sigmoid()
                edge_prob = edge_logit.sigmoid()
                vertex_prob = vertex_logit.sigmoid()

                result.update(dict(masks=mask_prob, edges=edge_prob, vertices=vertex_prob,
                                   polygons=pred_polygon, adjacency=pred_adjacent))

        return result, losses