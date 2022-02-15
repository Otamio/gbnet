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

np.random.seed(1234)
torch.manual_seed(1235)

codebase = './'
exp_name = 'exp_132'
eval_epoch = 10

synsetColor = ['white', 'green', 'gray', 'brown', 'blue', 'black', 'pink',
               'yellow', 'purple', 'red', 'grey', 'orange', 'pink', 'light brown', 'light blue']
synsetTexture = ['metal', 'plastic', 'wooden', 'glass', 'silver']

with open(pipeline.vgg_json_processed_path) as fd:
    metadata = json.load(fd)

with open("/nas/home/jiangwan/gbnet/ipynb/eval_sgdet/prob_label.json") as fd:
    prob_label = json.load(fd)
    prob_label = defaultdict(lambda: defaultdict(float), prob_label)


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
    ckpt = torch.load(conf.ckpt)
    ckptAttrNet = torch.load(pipeline.attrnet_params)

    # Load Images
    test = Images('test')

    # Loda Model
    gbNetEval = GBNetAttrEval(classes=test.ind_to_classes, rel_classes=test.ind_to_predicates,
                              num_gpus=conf.num_gpus, mode=conf.mode, require_overlap_det=True,
                              use_resnet=conf.use_resnet, use_proposals=conf.use_proposals,
                              pooling_dim=conf.pooling_dim,
                              ggnn_rel_time_step_num=3, ggnn_rel_hidden_dim=1024, ggnn_rel_output_dim=None,
                              graph_path=os.path.join(codebase, 'graphs/005/all_edges.pkl'),
                              emb_path=os.path.join(codebase, 'graphs/001/emb_mtx.pkl'),
                              rel_counts_path=os.path.join(codebase, 'graphs/001/pred_counts.pkl'),
                              use_knowledge=True, use_embedding=True, refine_obj_cls=False,
                              class_volume=1.0, top_k_to_keep=5, normalize_messages=False
                              )
    optimistic_restore(gbNetEval, ckpt['state_dict'])
    gbNetEval.attrNet[1].weight.data.copy_(ckptAttrNet['1.weight'])
    gbNetEval.attrNet[1].bias.data.copy_(ckptAttrNet['1.bias'])
    gbNetEval.attrNet[4].weight.data.copy_(ckptAttrNet['4.weight'])
    gbNetEval.attrNet[4].bias.data.copy_(ckptAttrNet['4.bias'])
    gbNetEval.attrNet[7].weight.data.copy_(ckptAttrNet['7.weight'])
    gbNetEval.attrNet[7].bias.data.copy_(ckptAttrNet['7.bias'])

    gbNetEval.cuda()
    gbNetEval.eval()

    # Generate the results
    results = {}

    for k in tqdm(range(len(test))):

        fname = test[k]['fn']

        test_objects = test[k]['gt_classes']
        test_attributes = test[k]['gt_attributes']
        test_attr_set = defaultdict(set)
        test_obj_set = set()
        test_attr_size = 0
        for obj, attr in test_attributes.nonzero():
            if metadata['idx_to_attribute'][str(attr.item() + 1)] == "green":
                continue
            test_obj_set.add(metadata['idx_to_label'][str(test_objects[obj.item()])])
            test_attr_set[metadata['idx_to_label'][str(test_objects[obj.item()])]].add(
                metadata['idx_to_attribute'][str(attr.item() + 1)])
            test_attr_size += 1

        image_id = os.path.basename(fname).split('.')[0]

        if os.path.exists(f'/nas/home/jiangwan/repo/gbnet/vg_out/{image_id}.csv'):
            continue

        test_slice = torch.utils.data.Subset(test, [k])
        _, image_loader = ImageDataLoader.splits(test_slice, test_slice, mode='rel',
                                                 batch_size=conf.batch_size,
                                                 num_workers=conf.num_workers,
                                                 num_gpus=conf.num_gpus)
        for batch in image_loader:
            batch.scatter()
            result = gbNetEval[batch]

        curr_attributes = defaultdict(list)
        top_object_cut = 0.5
        top_object_len = len([x.item() for x in (result.obj_scores > top_object_cut).nonzero()])
        top_k = 10

        for i, (obj, attrs_prob, attrs) in enumerate(zip(result.obj_preds, torch.topk(result.attr_pred, top_k, 1)[0],
                                                         torch.topk(result.attr_pred, top_k, 1)[1])):
            if i >= top_object_len:
                break
            obj_name = metadata['idx_to_label'][str(obj.item())]
            for attr_prob, attr_id in zip(attrs_prob[1:], attrs[1:]):
                attr_name = metadata['idx_to_attribute'][str(attr_id.item() + 1)]
                curr_attributes[(i, obj_name)].append((attr_name, attr_prob.item(), prob_label[obj_name][attr_name]))

        dict_frame = []

        for key, val in curr_attributes.items():

            num_colors = 0
            num_texture = 0
            small_big_large_little_flag = False
            open_close_flag = False
            walking_standing_flag = False
            short_long_flag = False
            thick_thin_flag = False

            for i in range(top_k - 1):

                # Filter rare attribute
                if val[i][2] < 0.003:
                    continue
                attr = val[i][0]

                # Load no more than 1 color
                if attr in synsetColor:
                    if num_colors > 1:
                        continue
                    num_colors += 1

                if attr in synsetTexture:
                    if num_texture >= 1:
                        continue
                    num_texture += 1

                if attr in ['open', 'closed']:
                    if open_close_flag:
                        continue
                    open_close_flag = True

                if attr in ['thick', 'thin']:
                    if thick_thin_flag:
                        continue
                    thick_thin_flag = True

                if attr in ['short', 'long']:
                    if short_long_flag:
                        continue
                    short_long_flag = True

                if attr in ['small', 'big', 'large', 'little']:
                    if small_big_large_little_flag:
                        continue
                    small_big_large_little_flag = True

                if attr in ['walking', 'standing', 'sitting']:
                    if walking_standing_flag:
                        continue
                    walking_standing_flag = True

                dict_frame.append({
                    'Id': key[0],
                    'Object': key[1],
                    'Attribute': val[i][0],
                    'Predicted Probability': val[i][1],
                    'VG Conditional Probability (Prior)': val[i][2],
                    'Probability Divide': val[i][1] / val[i][2] if val[i][2] >= 0.002 else 0
                })

            if len(dict_frame) == 0:
                df = pd.DataFrame(columns=['Object', 'Attribute', 'Predicted Probability',
                                           'VG Conditional Probability (Prior)', 'Probability Divide'])
            else:
                df = pd.DataFrame(dict_frame).set_index('Id')
                df = df.sort_values(by=['Id', 'Predicted Probability', 'Probability Divide'],
                                    ascending=[True, False, False])
                df = df[df['Probability Divide'] > 1.1]
                df['Hit'] = False
                # Find corresponding attributes
                obj_find = 0
                obj_total = len(test_obj_set)
                find = 0
                for i, row in df.iterrows():
                    if row['Object'] in test_obj_set:
                        test_obj_set.remove(row['Object'])
                        obj_find += 1
                    if row['Object'] in test_attr_set and row['Attribute'] in test_attr_set[row['Object']]:
                        test_attr_set[row['Object']].remove(row['Attribute'])
                        find += 1
                        df.iloc[i]['Hit'] = True
                results[fname] = (find, test_attr_size, obj_find, obj_total)

            df.to_csv(f'/nas/home/jiangwan/repo/gbnet/vg_out/{image_id}.csv', index=None, sep='\t')
