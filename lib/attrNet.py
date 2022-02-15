"""
from my_model_32: new ggnn
"""
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import PackedSequence
from lib.resnet import resnet_l4
from config import BATCHNORM_MOMENTUM
from lib.fpn.nms.functions.nms import apply_nms

from lib.fpn.box_utils import bbox_overlaps, center_size
from lib.fpn.proposal_assignments.rel_assignments import rel_assignments
from lib.object_detector import ObjectDetector, gather_res, load_vgg
from lib.pytorch_misc import transpose_packed_sequence_inds, onehot_logits, arange, enumerate_by_image, diagonal_inds, Flattener
from lib.surgery import filter_dets
from lib.fpn.roi_align.functions.roi_align import RoIAlignFunction
from lib.my_ggnn_16 import GGNN

np.set_printoptions(threshold=sys.maxsize)

MODES = ('sgdet', 'sgcls', 'predcls')


class AttrNet(nn.Module):
    def __init__(self, classes, rel_classes, graph_path, emb_path, mode='sgdet', num_gpus=1, 
                 require_overlap_det=True, pooling_dim=4096, use_resnet=False, thresh=0.01,
                 use_proposals=False,
                 ggnn_rel_time_step_num=3,
                 ggnn_rel_hidden_dim=512,
                 ggnn_rel_output_dim=512, use_knowledge=True, use_embedding=True, refine_obj_cls=False,
                 rel_counts_path=None, class_volume=1.0, top_k_to_keep=5, normalize_messages=True):

        """
        :param classes: Object classes
        :param rel_classes: Relationship classes. None if were not using rel mode
        :param mode: (sgcls, predcls, or sgdet)
        :param num_gpus: how many GPUS 2 use
        :param require_overlap_det: Whether two objects must intersect
        """
        super(AttrNet, self).__init__()
        self.classes = classes
        self.rel_classes = rel_classes
        self.num_gpus = num_gpus
        assert mode in MODES
        self.mode = mode
        self.pooling_size = 7
        self.obj_dim = 2048 if use_resnet else 4096
        self.rel_dim = self.obj_dim
        self.pooling_dim = pooling_dim

        self.require_overlap = require_overlap_det and self.mode == 'sgdet'

        self.detector = ObjectDetector(
            classes=classes,
            mode=('proposals' if use_proposals else 'refinerels') if mode == 'sgdet' else 'gtbox',
            use_resnet=use_resnet,
            thresh=thresh,
            max_per_img=64
        )

        if use_resnet:
            self.roi_fmap = nn.Sequential(
                resnet_l4(relu_end=False),
                nn.AvgPool2d(self.pooling_size),
                Flattener(),
            )
        else:
            roi_fmap = [
                Flattener(),
                load_vgg(use_dropout=False, use_relu=False, use_linear=pooling_dim == 4096, pretrained=False).classifier,
            ]
            if pooling_dim != 4096:
                roi_fmap.append(nn.Linear(4096, pooling_dim))
            self.roi_fmap = nn.Sequential(*roi_fmap)
            self.roi_fmap_obj = load_vgg(pretrained=False).classifier
        
        self.attrNet = torch.nn.Sequential(
            torch.nn.Dropout(0.2),
            torch.nn.Linear(self.obj_dim, 512),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(512, 256),
            torch.nn.Dropout(0.3),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 51),
            torch.nn.Dropout(0.3),
            torch.nn.Sigmoid()
        )
        
        self.loss_fn = torch.nn.BCELoss()
        self.optimizer = torch.optim.Adam(self.attrNet.parameters(), weight_decay=0.0, lr=0.0001, eps=1e-3)

    @property
    def num_classes(self):
        return len(self.classes)

    @property
    def num_rels(self):
        return len(self.rel_classes)

    def obj_feature_map(self, features, rois):
        """
        Gets the ROI features
        :param features: [batch_size, dim, IM_SIZE/4, IM_SIZE/4] (features at level p2)
        :param rois: [num_rois, 5] array of [img_num, x0, y0, x1, y1].
        :return: [num_rois, #dim] array
        """
        feature_pool = RoIAlignFunction(self.pooling_size, self.pooling_size, spatial_scale=1 / 16)(
            features, rois)
        return self.roi_fmap_obj(feature_pool.view(rois.size(0), -1))

    def forward(self, x, im_sizes, image_offset,
                gt_boxes=None, gt_classes=None, gt_rels=None, proposals=None, train_anchor_inds=None,
                gt_attributes=None, fnames=None, return_fmap=False):
        """
        Forward pass for detection
        :param x: Images@[batch_size, 3, IM_SIZE, IM_SIZE]
        :param im_sizes: A numpy array of (h, w, scale) for each image.
        :param image_offset: Offset onto what image we're on for MGPU training (if single GPU this is 0)
        :param gt_boxes:
        Training parameters:
        :param gt_boxes: [num_gt, 4] GT boxes over the batch.
        :param gt_classes: [num_gt, 2] gt boxes where each one is (img_id, class)
        :param train_anchor_inds: a [num_train, 2] array of indices for the anchors that will
                                  be used to compute the training loss. Each (img_ind, fpn_idx)
        :return: If train:
            scores, boxdeltas, labels, boxes, boxtargets, rpnscores, rpnboxes, rellabels
            
            if test:
            prob dists, boxes, img inds, maxscores, classes
            
        """
        result = self.detector(x, im_sizes, image_offset, gt_boxes, gt_classes, gt_rels, proposals,
                               train_anchor_inds, return_fmap=True)
        if result.is_none():
            return ValueError("heck")

        im_inds = result.im_inds - image_offset
        boxes = result.rm_box_priors

        rois = torch.cat((im_inds[:, None].float(), boxes), 1)

        result.obj_fmap = self.obj_feature_map(result.fmap.detach(), rois)
                        
        result.attr_pred = self.attrNet(result.obj_fmap)
        result.attr_gt = gt_attributes
        result.fnames = fnames

        return result

    def __getitem__(self, batch):
        """ Hack to do multi-GPU training"""
        batch.scatter()
        if self.num_gpus == 1:
            return self(*batch[0])
        replicas = nn.parallel.replicate(self, devices=list(range(self.num_gpus)))
        outputs = nn.parallel.parallel_apply(replicas, [batch[i] for i in range(self.num_gpus)])
        if self.training:
            return gather_res(outputs, 0, dim=0)
        return outputs
    
    def loss_attr(self, result):
        result.attr_gt = result.attr_gt.cuda()
        return self.loss_fn(result.attr_pred, result.attr_gt)
