import os
import numpy as np
import sys
import torch

sys.path.append('./')
import pipeline
from config import ModelConfig
from lib.pytorch_misc import optimistic_restore
from lib.attrNet import AttrNet
from dataloaders.visual_genome_ext import Images, ImageDataLoader
from tqdm import tqdm

import json

np.random.seed(1234)
torch.manual_seed(1235)

exp_name = 'exp_132'


def get_config():
    return ModelConfig(f'''
            -m predcls -p 1000 -clip 5 
            -tb_log_dir summaries/kern_predcls/{exp_name} 
            -save_dir checkpoints/kern_predcls/{exp_name}
            -ckpt checkpoints/vgdet/vg-24.tar 
            -val_size 5000 
            -adam 
            -b 4
            -ngpu 1
            -lr 1e-4 
            ''')


def get_attrnet(train, conf, codebase='./'):
    attrnet = AttrNet(classes=train.ind_to_classes, rel_classes=train.ind_to_predicates,
                      num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                      use_resnet=conf.use_resnet, use_proposals=conf.use_proposals, pooling_dim=conf.pooling_dim,
                      ggnn_rel_time_step_num=3, ggnn_rel_hidden_dim=1024, ggnn_rel_output_dim=None,
                      graph_path=os.path.join(codebase, 'graphs/005/all_edges.pkl'),
                      emb_path=os.path.join(codebase, 'graphs/001/emb_mtx.pkl'),
                      rel_counts_path=os.path.join(codebase, 'graphs/001/pred_counts.pkl'),
                      use_knowledge=True, use_embedding=True, refine_obj_cls=False,
                      class_volume=1.0, top_k_to_keep=5, normalize_messages=False,
                      )
    return attrnet


def main():

    conf = get_config()
    train = Images('train')  # , num_im=500, num_val_im=100)  # Just pull the first 100 elements for quick testing
    val = Images('val')  #, num_im=500, num_val_im=100)
    ind_to_predicates = train.ind_to_predicates  # ind_to_predicates[0] means no relationship

    train_loader, val_loader = ImageDataLoader.splits(train, val, mode='rel',
                                                      batch_size=conf.batch_size,
                                                      num_workers=conf.num_workers,
                                                      num_gpus=conf.num_gpus)
    # Restore Attribute Data
    ckpt = torch.load(conf.ckpt)
    attrnet = get_attrnet(train, conf)
    optimistic_restore(attrnet, ckpt['state_dict'])

    # Freeze Parameters
    attrnet.cuda()

    # Freeze gradients
    for n, param in attrnet.detector.named_parameters():
        param.requires_grad = False
    for n, param in attrnet.roi_fmap.named_parameters():
        param.requires_grad = False
    for n, param in attrnet.roi_fmap_obj.named_parameters():
        param.requires_grad = False

    with open(pipeline.vgg_json_processed_path) as fd:
        vgg = json.load(fd)

    print()
    print('######################')
    print('Training starts now!!!')
    attrnet.train()

    for epoch in tqdm(range(1, 200)):
        for batch in tqdm(train_loader):
            batch.scatter()
            result = attrnet[batch]

            loss = attrnet.loss_attr(result)

            attrnet.optimizer.zero_grad()

            loss.backward()

            attrnet.optimizer.step()

        print(f'Loss at epoch {epoch}', loss.item())
        if epoch % 5 == 0:
            torch.save(attrnet.attrNet.state_dict(), pipeline.attrnet_params)
