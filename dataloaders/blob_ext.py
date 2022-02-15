"""
Data blob, hopefully to make collating less painful and MGPU training possible
"""
from lib.fpn.anchor_targets import anchor_target_layer
import numpy as np
import torch
from torch.autograd import Variable

from dataloaders.blob import Blob

class Image_Batch(Blob):
    def __init__(self, mode='det', is_train=False, num_gpus=1, primary_gpu=0, batch_size_per_gpu=3, dataset='VG'):
        """
        Initializes an empty Blob object.
        :param mode: 'det' for detection and 'rel' for det+relationship
        :param is_train: True if it's training
        """
        super().__init__(mode, is_train, num_gpus, primary_gpu, batch_size_per_gpu)
        self._dataset = dataset
        self.gt_attributes = []
        self.fnames = []
        self.index = []

    @property
    def dataset(self):
        return self.dataset

    def append(self, d):
        """
        Adds a single image to the blob
        :param datom:
        :return:
        """
        if d['dataset'].lower() == 'vg':
            super().append(d);
        elif d['dataset'].lower() == 'awa2':
            self.imgs.append(d['img'])
            self.fnames.append(d['fn'])
            h, w, scale = d['img_size']
            self.im_sizes.append((h, w, scale))
            
        self.gt_attributes.append(d['gt_attributes'])
        self.index.append(d['index'])

    def _chunkize(self, datom, tensor=torch.LongTensor):
        """
        Turn data list into chunks, one per GPU
        :param datom: List of lists of numpy arrays that will be concatenated.
        :return:
        """
        chunk_sizes = [0] * self.num_gpus
        for i in range(self.num_gpus):
            for j in range(self.batch_size_per_gpu):
                chunk_sizes[i] += datom[i * self.batch_size_per_gpu + j].shape[0]
        return Variable(tensor(np.concatenate(datom, 0)), volatile=self.volatile), chunk_sizes

    def reduce(self):
        """ Merges all the detections into flat lists + numbers of how many are in each"""
        if len(self.imgs) != self.batch_size_per_gpu * self.num_gpus:
            raise ValueError("Wrong batch size? imgs len {} bsize/gpu {} numgpus {}".format(
                len(self.imgs), self.batch_size_per_gpu, self.num_gpus
            ))

        self.imgs = Variable(torch.stack(self.imgs, 0), volatile=self.volatile)
        self.im_sizes = np.stack(self.im_sizes).reshape(
            (self.num_gpus, self.batch_size_per_gpu, 3))

        if self.is_rel and self.gt_rels:
            self.gt_rels, self.gt_rel_chunks = self._chunkize(self.gt_rels)
        else:
            self.gt_rels = torch.tensor([], dtype=torch.int32)
            self.gt_rel_chunks = torch.tensor([], dtype=torch.int32)
        
        if self.gt_boxes:
            self.gt_boxes, self.gt_box_chunks = self._chunkize(self.gt_boxes, tensor=torch.FloatTensor)
        else:
            self.gt_boxes = torch.tensor([], dtype=torch.int32)
            self.gt_box_chunks = torch.tensor([], dtype=torch.int32)
        
        if self.gt_classes:
            self.gt_classes, _ = self._chunkize(self.gt_classes)
        else:
            self.gt_classes = torch.tensor([], dtype=torch.int32)
        
        if self.gt_attributes:
            self.gt_attributes, _ = self._chunkize(self.gt_attributes, tensor=torch.FloatTensor)
            
        if self.is_train:
            self.train_anchor_labels, self.train_chunks = self._chunkize(self.train_anchor_labels)
            self.train_anchors, _ = self._chunkize(self.train_anchors, tensor=torch.FloatTensor)
            self.train_anchor_inds = self.train_anchor_labels[:, :-1].contiguous()

        if len(self.proposals) != 0:
            self.proposals, self.proposal_chunks = self._chunkize(self.proposals, tensor=torch.FloatTensor)

    def __getitem__(self, index):
        """
        Returns a tuple containing data
        :param index: Which GPU we're on, or 0 if no GPUs
        :return: If training:
        (image, im_size, img_start_ind, anchor_inds, anchors, gt_boxes, gt_classes, 
        train_anchor_inds)
        test:
        (image, im_size, img_start_ind, anchor_inds, anchors)
        """
        if index not in list(range(self.num_gpus)):
            raise ValueError("Out of bounds with index {} and {} gpus".format(index, self.num_gpus))

        if self.is_rel:
            rels = self.gt_rels
            if index > 0 or self.num_gpus != 1:
                rels_i = rels[index] if self.is_rel else None
        elif self.is_flickr:
            rels = (self.gt_sents, self.gt_nodes)
            if index > 0 or self.num_gpus != 1:
                rels_i = (self.gt_sents[index], self.gt_nodes[index])
        else:
            rels = None
            rels_i = None

        if self.proposal_chunks is None:
            proposals = None
        else:
            proposals = self.proposals
        
        if index == 0 and self.num_gpus == 1:
            image_offset = 0
            if self.is_train:
                return (self.imgs, self.im_sizes[0], image_offset,
                        self.gt_boxes, self.gt_classes, rels, proposals, self.train_anchor_inds, self.gt_attributes, self.fnames)
            return self.imgs, self.im_sizes[0], image_offset, self.gt_boxes, self.gt_classes, rels, proposals, self.gt_attributes, self.fnames

        # Otherwise proposals is None
        assert proposals is None

        image_offset = self.batch_size_per_gpu * index
        # TODO: Return a namedtuple
        if self.is_train:
            return (
            self.imgs[index], self.im_sizes[index], image_offset,
            self.gt_boxes[index], self.gt_classes[index], rels_i, None, self.train_anchor_inds[index], self.gt_attributes[index], self.fnames[index])
        return (self.imgs[index], self.im_sizes[index], image_offset,
                self.gt_boxes[index], self.gt_classes[index], rels_i, None, self.gt_attributes[index], self.fnames[index])