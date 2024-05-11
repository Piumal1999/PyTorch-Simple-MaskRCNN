from collections import OrderedDict

import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.model_zoo import load_url
from torchvision import models
from torchvision.models.detection import backbone_utils, _utils
from torchvision.ops import misc
from torchvision.ops import MultiScaleRoIAlign

from .utils import AnchorGenerator
from .rpn import RPNHead, RegionProposalNetwork
from .pooler import RoIAlign
from .roi_heads import RoIHeads
from .transform import Transformer

from .GNN.poly_gnn import PolyGNN


class MaskRCNN(nn.Module):
    """
    Implements Mask R-CNN.

    The input image to the model is expected to be a tensor, shape [C, H, W], and should be in 0-1 range.

    The behavior of the model changes depending if it is in training or evaluation mode.

    During training, the model expects both the input tensor, as well as a target (dictionary),
    containing:
        - boxes (FloatTensor[N, 4]): the ground-truth boxes in [xmin, ymin, xmax, ymax] format, with values
          between 0-H and 0-W
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - masks (UInt8Tensor[N, H, W]): the segmentation binary masks for each instance

    The model returns a Dict[Tensor], containing the classification and regression losses
    for both the RPN and the R-CNN, and the mask loss.

    During inference, the model requires only the input tensor, and returns the post-processed
    predictions as a Dict[Tensor]. The fields of the Dict are as
    follows:
        - boxes (FloatTensor[N, 4]): the predicted boxes in [xmin, ymin, xmax, ymax] format,
          with values between 0-H and 0-W
        - labels (Int64Tensor[N]): the predicted labels
        - scores (FloatTensor[N]): the scores for each prediction
        - masks (FloatTensor[N, H, W]): the predicted masks for each instance, in 0-1 range. In order to
          obtain the final segmentation masks, the soft masks can be thresholded, generally
          with a value of 0.5 (mask >= 0.5)

    Arguments:
        backbone (nn.Module): the network used to compute the features for the model.
        num_classes (int): number of output classes of the model (including the background).

        rpn_fg_iou_thresh (float): minimum IoU between the anchor and the GT box so that they can be
            considered as positive during training of the RPN.
        rpn_bg_iou_thresh (float): maximum IoU between the anchor and the GT box so that they can be
            considered as negative during training of the RPN.
        rpn_num_samples (int): number of anchors that are sampled during training of the RPN
            for computing the loss
        rpn_positive_fraction (float): proportion of positive anchors during training of the RPN
        rpn_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
        rpn_pre_nms_top_n_train (int): number of proposals to keep before applying NMS during training
        rpn_pre_nms_top_n_test (int): number of proposals to keep before applying NMS during testing
        rpn_post_nms_top_n_train (int): number of proposals to keep after applying NMS during training
        rpn_post_nms_top_n_test (int): number of proposals to keep after applying NMS during testing
        rpn_nms_thresh (float): NMS threshold used for postprocessing the RPN proposals

        box_fg_iou_thresh (float): minimum IoU between the proposals and the GT box so that they can be
            considered as positive during training of the classification head
        box_bg_iou_thresh (float): maximum IoU between the proposals and the GT box so that they can be
            considered as negative during training of the classification head
        box_num_samples (int): number of proposals that are sampled during training of the
            classification head
        box_positive_fraction (float): proportion of positive proposals during training of the
            classification head
        box_reg_weights (Tuple[float, float, float, float]): weights for the encoding/decoding of the
            bounding boxes
        box_score_thresh (float): during inference, only return proposals with a classification score
            greater than box_score_thresh
        box_nms_thresh (float): NMS threshold for the prediction head. Used during inference
        box_num_detections (int): maximum number of detections, for all classes.

    """

    def __init__(self, backbone, num_classes,
                 # RPN parameters
                 rpn_fg_iou_thresh=0.7, rpn_bg_iou_thresh=0.3,
                 rpn_num_samples=256, rpn_positive_fraction=0.5,
                 rpn_reg_weights=(1., 1., 1., 1.),
                 rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
                 rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
                 rpn_nms_thresh=0.7,
                 # RoIHeads parameters
                 box_fg_iou_thresh=0.5, box_bg_iou_thresh=0.5,
                 box_num_samples=512, box_positive_fraction=0.25,
                 box_reg_weights=(10., 10., 5., 5.),
                 box_score_thresh=0.1, box_nms_thresh=0.6, box_num_detections=100):
        super().__init__()
        self.backbone = backbone
        out_channels = backbone.out_channels

        #------------- RPN --------------------------
        anchor_sizes = (32, 64, 128, 256, 512)
        anchor_ratios = (0.5, 1, 2)
        num_anchors = len(anchor_sizes) * len(anchor_ratios)
        rpn_anchor_generator = AnchorGenerator(anchor_sizes, anchor_ratios)
        rpn_head = RPNHead(out_channels, num_anchors)

        rpn_pre_nms_top_n = dict(training=rpn_pre_nms_top_n_train, testing=rpn_pre_nms_top_n_test)
        rpn_post_nms_top_n = dict(training=rpn_post_nms_top_n_train, testing=rpn_post_nms_top_n_test)
        self.rpn = RegionProposalNetwork(
             rpn_anchor_generator, rpn_head,
             rpn_fg_iou_thresh, rpn_bg_iou_thresh,
             rpn_num_samples, rpn_positive_fraction,
             rpn_reg_weights,
             rpn_pre_nms_top_n, rpn_post_nms_top_n, rpn_nms_thresh)

        #------------ RoIHeads --------------------------
        box_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=7, sampling_ratio=2)
        resolution = box_roi_pool.output_size[0]
        in_channels = out_channels * resolution ** 2
        mid_channels = 1024
        box_predictor = FastRCNNPredictor(in_channels, mid_channels, num_classes)

        self.head = RoIHeads(box_roi_pool, box_predictor,
                             box_fg_iou_thresh, box_bg_iou_thresh,
                             box_num_samples, box_positive_fraction,
                             box_reg_weights, box_score_thresh,
                             box_nms_thresh, box_num_detections)

        self.head.mask_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"], output_size=14, sampling_ratio=2)
        self.head.augmentation_roi_pool = MultiScaleRoIAlign(featmap_names=["0"], output_size=28, sampling_ratio=2)
        layers_mask = (256, 256, 256, 256)
        dim_reduced_mask = 256

        self.head.mask_predictor = MaskRCNNPredictor(out_channels, layers_mask, dim_reduced_mask, num_classes)
        self.head.feature_augmentor = FeatureAugmentor(feats_dim=28, feats_channels=256, internal=2)
        self.head.poly_augmentor = PolyAugmentor(out_channels)
        self.head.polygon_predictor = PolyGNN(out_channels, feature_grid_size=28)

        # ------------ Transformer --------------------------
        self.transformer = Transformer(
            min_size=800, max_size=1333,
            image_mean=[0.339, 0.327, 0.329],  # ImageNet: image_mean=[0.485, 0.456, 0.406],
            image_std=[0.255, 0.246, 0.242])   # ImageNet: image_std=[0.229, 0.224, 0.225]

        # TODO: transformer changes mask too! so, include edge&vertex too.

    def forward(self, image, target=None):
        ori_image_shape = image.shape[-2:]

        image, target = self.transformer(image, target)
        image_shape = image.shape[-2:]
        feature = self.backbone(image)
        proposal = []
        rpn_losses = {'rpn_objectness_loss': 0.0, 'rpn_box_loss': 0.0}
        for k in feature.keys():
            prop, rpn_loss = self.rpn(feature[k], image_shape, target)
            proposal.append(prop)
            if 'rpn_objectness_loss' in rpn_loss:
                rpn_losses['rpn_objectness_loss'] += rpn_loss['rpn_objectness_loss']
                rpn_losses['rpn_box_loss'] += rpn_loss['rpn_box_loss']

        if rpn_losses['rpn_objectness_loss']:
            rpn_losses['rpn_objectness_loss'] /= 5
            rpn_losses['rpn_box_loss'] /= 5
        
        proposal = torch.vstack(proposal)
        result, roi_losses = self.head(feature, proposal, image_shape, target)

        if self.training:
            return dict(**rpn_losses, **roi_losses)
        else:
            result = self.transformer.postprocess(result, image_shape, ori_image_shape)
            return result


class FastRCNNPredictor(nn.Module):
    def __init__(self, in_channels, mid_channels, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(in_channels, mid_channels)
        self.fc2 = nn.Linear(mid_channels, mid_channels)
        self.cls_score = nn.Linear(mid_channels, num_classes)
        self.bbox_pred = nn.Linear(mid_channels, num_classes * 4)

    def forward(self, x):
        x = x.flatten(start_dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        score = self.cls_score(x)
        bbox_delta = self.bbox_pred(x)

        return score, bbox_delta


class MaskRCNNPredictor(nn.Sequential):
    def __init__(self, in_channels, layers, dim_reduced, num_classes):
        """
        Arguments:
            in_channels (int)
            layers (Tuple[int])
            dim_reduced (int)
            num_classes (int)
        """

        d = OrderedDict()
        next_feature = in_channels
        for layer_idx, layer_features in enumerate(layers, 1):
            d['mask_fcn{}'.format(layer_idx)] = nn.Conv2d(next_feature, layer_features, 3, 1, 1)
            d['mask_relu{}'.format(layer_idx)] = nn.ReLU(inplace=True)
            next_feature = layer_features

        d['mask_conv_trans'] = nn.ConvTranspose2d(next_feature, dim_reduced, 2, 2, 0)
        d['mask_relu_trans'] = nn.ReLU(inplace=True)
        d['mask_fcn_logits'] = nn.Conv2d(dim_reduced, num_classes, 1, 1, 0)
        super().__init__(d)

        for name, param in self.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param, mode='fan_out', nonlinearity='relu')

class FeatureAugmentor(nn.Module):
    def __init__(self, feats_dim=28, feats_channels=256, internal=2):
        super(FeatureAugmentor, self).__init__()
        self.grid_size = feats_dim

        self.edge_conv = nn.Conv2d(in_channels=feats_channels,
                                   out_channels=internal,
                                   kernel_size=3,
                                   padding=1)

        self.edge_fc = nn.Linear(in_features=feats_dim**2 * internal,
                                 out_features=feats_dim**2)

        self.vertex_conv = nn.Conv2d(in_channels=feats_channels,
                                     out_channels=internal,
                                     kernel_size=3,
                                     padding=1)

        self.vertex_fc = nn.Linear(in_features=feats_dim**2 * internal,
                                   out_features=feats_dim**2)

    def forward(self, feats, temperature=0.0, beam_size=1):
        """
        if temperature < 0.01, use greedy
        else, use temperature
        """
        batch_size = feats.size(0)
        conv_edge = self.edge_conv(feats)
        conv_edge = F.relu(conv_edge, inplace=True)
        edge_logits = self.edge_fc(conv_edge.view(batch_size, -1))

        conv_vertex = self.vertex_conv(feats)
        conv_vertex = F.relu(conv_vertex)
        vertex_logits = self.vertex_fc(conv_vertex.view(batch_size, -1))

        edge_logits = edge_logits.view((-1, self.grid_size, self.grid_size)).unsqueeze(1)
        vertex_logits = vertex_logits.view((-1, self.grid_size, self.grid_size)).unsqueeze(1)

        return edge_logits, vertex_logits


class PolyAugmentor(nn.Sequential):
    def __init__(self, in_channels):
        """
        Equivalent to Edge Annotation in Kang's
        Arguments:
            in_channels (int)
        """

        d = OrderedDict()

        d['cnn1'] = nn.Conv2d(in_channels + 2, 256, 3, 1, 1, bias=False)
        d['bn1'] = nn.BatchNorm2d(256)
        d['relu1'] = nn.ReLU(inplace=True)

        d['cnn2'] = nn.Conv2d(256, 256, 3, 1, 1, bias=False)
        d['bn2'] = nn.BatchNorm2d(256)
        d['relu2'] = nn.ReLU(inplace=True)

        super().__init__(d)


class ResBackbone(nn.Module):
    def __init__(self, backbone_name, pretrained):
        super().__init__()
        body = models.resnet.__dict__[backbone_name](
            pretrained=pretrained, norm_layer=misc.FrozenBatchNorm2d)

        for name, parameter in body.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)

        self.body = nn.ModuleDict(d for i, d in enumerate(body.named_children()) if i < 8)
        in_channels = 2048
        self.out_channels = 256

        self.inner_block_module = nn.Conv2d(in_channels, self.out_channels, 1)
        self.layer_block_module = nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1)

        for m in self.children():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        for module in self.body.values():
            x = module(x)
        x = self.inner_block_module(x)
        x = self.layer_block_module(x)
        return x


def maskrcnn_resnet50(pretrained, num_classes, pretrained_backbone=True):
    """
    Constructs a Mask R-CNN model with a ResNet-50 backbone.

    Arguments:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017.
        num_classes (int): number of classes (including the background).
    """
    if pretrained:
        pretrained_backbone = False

    backbone = ResBackbone('resnet50', pretrained_backbone)
    model = MaskRCNN(backbone, num_classes)

    if pretrained:
        model_urls = {
            'maskrcnn_resnet50_fpn_coco':
                'https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth',
        }
        model_state_dict = load_url(model_urls['maskrcnn_resnet50_fpn_coco'])

        pretrained_msd = list(model_state_dict.values())
        del_list = [i for i in range(265, 271)] + [i for i in range(273, 279)]
        for i, del_idx in enumerate(del_list):
            pretrained_msd.pop(del_idx - i)

        msd = model.state_dict()
        skip_list = [271, 272, 273, 274, 279, 280, 281, 282, 293, 294]
        if num_classes == 91:
            skip_list = [271, 272, 273, 274]
        for i, name in enumerate(msd):
            if i in skip_list:
                continue
            msd[name].copy_(pretrained_msd[i])

        model.load_state_dict(msd)

    return model


def maskrcnn_resnet_fpn(
    pretrained=False, progress=True, num_classes=2, pretrained_backbone=True, 
    trainable_backbone_layers=None, backbone_name='resnet50', **kwargs):
    """
    Constructs a Mask R-CNN model with a ResNet-50/101-FPN backbone.
    Reference: `"Mask R-CNN" <https://arxiv.org/abs/1703.06870>`_.
    The input to the model is expected to be a list of tensors, each of shape ``[C, H, W]``, one for each
    image, and should be in ``0-1`` range. Different images can have different sizes.
    The behavior of the model changes depending if it is in training or evaluation mode.
    During training, the model expects both the input tensors, as well as a targets (list of dictionary),
    containing:
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the class label for each ground-truth box
        - masks (``UInt8Tensor[N, H, W]``): the segmentation binary masks for each instance
    The model returns a ``Dict[Tensor]`` during training, containing the classification and regression
    losses for both the RPN and the R-CNN, and the mask loss.
    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a ``List[Dict[Tensor]]``, one for each input image. The fields of the ``Dict`` are as
    follows, where ``N`` is the number of detected instances:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format, with
          ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (``Int64Tensor[N]``): the predicted labels for each instance
        - scores (``Tensor[N]``): the scores or each instance
        - masks (``UInt8Tensor[N, 1, H, W]``): the predicted masks for each instance, in ``0-1`` range. In order to
          obtain the final segmentation masks, the soft masks can be thresholded, generally
          with a value of 0.5 (``mask >= 0.5``)
    For more details on the output and on how to plot the masks, you may refer to :ref:`instance_seg_output`.
    Mask R-CNN is exportable to ONNX for a fixed batch size with inputs images of fixed size.
    Example::
        >>> model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
        >>> model.eval()
        >>> x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
        >>> predictions = model(x)
        >>>
        >>> # optionally, if you want to export the model to ONNX:
        >>> torch.onnx.export(model, x, "mask_rcnn.onnx", opset_version = 11)
    Args:
        pretrained (bool): If True, returns a model pre-trained on COCO train2017
        progress (bool): If True, displays a progress bar of the download to stderr
        num_classes (int): number of output classes of the model (including the background)
        pretrained_backbone (bool): If True, returns a model with backbone pre-trained on Imagenet
        trainable_backbone_layers (int): number of trainable (not frozen) resnet layers starting from final block.
            Valid values are between 0 and 5, with 5 meaning all backbone layers are trainable. If ``None`` is
            passed (the default) this value is set to 3.
    """
    model_urls = {
    "maskrcnn_resnet50_fpn_coco": "https://download.pytorch.org/models/maskrcnn_resnet50_fpn_coco-bf2d0c1e.pth",
    "maskrcnn_resnet101_fpn_coco": "https://download.pytorch.org/models/maskrcnn_resnet101_fpn_coco-bf2d0c1e.pth",
    }

    backbone = backbone_utils.resnet_fpn_backbone(backbone_name, pretrained=pretrained_backbone, norm_layer=misc.FrozenBatchNorm2d,
                                                  trainable_layers=3, returned_layers=None, extra_blocks=None)
    model = MaskRCNN(backbone, num_classes, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["maskrcnn_" + backbonename + "_fpn_coco"], progress=progress)
        model.load_state_dict(state_dict)
        _utils.overwrite_eps(model, 0.0)
    return model