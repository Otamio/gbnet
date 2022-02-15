import os
import numpy as np
import sys
import torch

sys.path.append('./')
from config import ModelConfig, BOX_SCALE, IM_SCALE
from lib.pytorch_misc import optimistic_restore

from tqdm import tqdm
import pickle
from dataloaders.visual_genome import VGDataLoader, VG
from lib.my_model_33 import KERN
from lib.evaluation.sg_eval import BasicSceneGraphEvaluator, calculate_mR_from_evaluator_list, eval_entry

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
    print('### Initializing dataloaders ###')
    train_loader, val_loader = VGDataLoader.splits(train, val, mode='rel',
                                                   batch_size=conf.batch_size,
                                                   num_workers=conf.num_workers,
                                                   num_gpus=conf.num_gpus)

    # Get the Object Detector and load parameters
    print('### Initializing detectors ###')
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

    # Begin Evaluation
    all_pred_entries = []
    print('Begin evaluation......')

    def val_batch(batch_num, b, evaluator, evaluator_multiple_preds, evaluator_list, evaluator_multiple_preds_list, thrs=(20, 50, 100)):
        det_res = detector[b]
        if conf.num_gpus == 1:
            det_res = [det_res]

        for i, (boxes_i, objs_i, obj_scores_i, rels_i, pred_scores_i) in enumerate(det_res):
            gt_entry = {
                'gt_classes': val.gt_classes[batch_num + i].copy(),
                'gt_relations': val.relationships[batch_num + i].copy(),
                'gt_boxes': val.gt_boxes[batch_num + i].copy(),
            }
            assert np.all(objs_i[rels_i[:,0]] > 0) and np.all(objs_i[rels_i[:,1]] > 0)
            # assert np.all(rels_i[:,2] > 0)

            pred_entry = {
                'pred_boxes': boxes_i * BOX_SCALE/IM_SCALE,
                'pred_classes': objs_i,
                'pred_rel_inds': rels_i,
                'obj_scores': obj_scores_i,
                'rel_scores': pred_scores_i,
            }
            all_pred_entries.append(pred_entry)

            eval_entry(conf.mode, gt_entry, pred_entry, evaluator, evaluator_multiple_preds,
                       evaluator_list, evaluator_multiple_preds_list)

    evaluator = BasicSceneGraphEvaluator.all_modes()
    evaluator_multiple_preds = BasicSceneGraphEvaluator.all_modes(multiple_preds=True)
    evaluator_list = []  # for calculating recall of each relationship except no relationship
    evaluator_multiple_preds_list = []
    for index, name in enumerate(ind_to_predicates):
        if index == 0:
            continue
        evaluator_list.append((index, name, BasicSceneGraphEvaluator.all_modes()))
        evaluator_multiple_preds_list.append((index, name, BasicSceneGraphEvaluator.all_modes(multiple_preds=True)))

    ########## IMPORTANT ############
    if conf.cache is not None and os.path.exists(conf.cache):
        print("Found {}! Loading from it".format(conf.cache))
        with open(conf.cache, 'rb') as f:
            all_pred_entries = pickle.load(f)
        for i, pred_entry in enumerate(tqdm(all_pred_entries)):
            gt_entry = {
                'gt_classes': val.gt_classes[i].copy(),
                'gt_relations': val.relationships[i].copy(),
                'gt_boxes': val.gt_boxes[i].copy(),
            }

            eval_entry(conf.mode, gt_entry, pred_entry, evaluator, evaluator_multiple_preds,
                       evaluator_list, evaluator_multiple_preds_list)

        recall = evaluator[conf.mode].print_stats()
        recall_mp = evaluator_multiple_preds[conf.mode].print_stats()

        mean_recall = calculate_mR_from_evaluator_list(evaluator_list, conf.mode, save_file=conf.save_rel_recall)
        mean_recall_mp = calculate_mR_from_evaluator_list(evaluator_multiple_preds_list, conf.mode, multiple_preds=True,
                                                          save_file=conf.save_rel_recall)

    else:
        detector.eval()
        for val_b, batch in enumerate(tqdm(val_loader)):
            val_batch(conf.num_gpus * val_b, batch, evaluator, evaluator_multiple_preds, evaluator_list,
                      evaluator_multiple_preds_list)

        recall = evaluator[conf.mode].print_stats()
        recall_mp = evaluator_multiple_preds[conf.mode].print_stats()

        mean_recall = calculate_mR_from_evaluator_list(evaluator_list, conf.mode, save_file=conf.save_rel_recall)
        mean_recall_mp = calculate_mR_from_evaluator_list(evaluator_multiple_preds_list, conf.mode, multiple_preds=True,
                                                          save_file=conf.save_rel_recall)

        if conf.cache is not None:
            with open(conf.cache, 'wb') as f:
                pickle.dump(all_pred_entries, f)
