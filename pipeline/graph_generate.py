import os
import numpy as np
import pandas as pd
import sys
import torch

sys.path.append('./')
import pipeline
from config import ModelConfig
from lib.pytorch_misc import optimistic_restore
from lib.GBNetAttrEval import GBNetAttrEval
from dataloaders.visual_genome_ext import Images, ImageDataLoader
from tqdm import tqdm
from collections import defaultdict

import json
from dataloaders.visual_genome import VGDataLoader, VG
from lib.my_model_33 import KERN

np.random.seed(1234)
torch.manual_seed(1235)

codebase = './'
exp_name = 'exp_132'
eval_epoch = 10


def get_config():
    return ModelConfig(f'''
    -m sgdet -p 1000 -clip 5 
    -ckpt checkpoints/kern_predcls/{exp_name}/vgrel-{eval_epoch}.tar 
    -test
    -b 1
    -ngpu 1
    -cache caches/{exp_name}/kern_sgdet-{eval_epoch}.pkl \
    -save_rel_recall results/{exp_name}/kern_rel_recall_sgdet-{eval_epoch}.pkl
    ''')

def main():

    # Load configuration
    conf = get_config()

    # Load datasets
    train, val, test = VG.splits(num_val_im=conf.val_size, filter_duplicate_rels=True,
                                 use_proposals=conf.use_proposals,
                                 filter_non_overlap=conf.mode == 'sgdet')
    ind_to_predicates = train.ind_to_predicates  # ind_to_predicates[0] means no relationship
    if conf.test:
        val = test

    # Pump the data into dataloaders
    train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                                   batch_size=conf.batch_size,
                                                   num_workers=conf.num_workers,
                                                   num_gpus=conf.num_gpus)

    # Get the Object Detector and load parameters
    detector = KERN(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
                    num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                    use_resnet=conf.use_resnet, use_proposals=conf.use_proposals, pooling_dim=conf.pooling_dim,
                    ggnn_rel_time_step_num=3, ggnn_rel_hidden_dim=1024, ggnn_rel_output_dim=None,
                    graph_path=os.path.join(codebase, 'graphs/005/all_edges.pkl'),
                    emb_path=os.path.join(codebase, 'graphs/001/emb_mtx.pkl'),
                    rel_counts_path=os.path.join(codebase, 'graphs/001/pred_counts.pkl'),
                    use_knowledge=True, use_embedding=True, refine_obj_cls=False,
                    class_volume=1.0, top_k_to_keep=5, normalize_messages=False,
                    )
    detector.cuda()
    ckpt = torch.load(conf.ckpt)
    optimistic_restore(detector, ckpt['state_dict'])


