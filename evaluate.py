import json
import lzma
import argparse

from dataset import ImagenetVidVRD, VidOR
from evaluation import eval_video_object, eval_video_relation


def evaluate_object(dataset, split, prediction):
    groundtruth = dict()
    for vid in dataset.get_index(split):
        groundtruth[vid] = dataset.get_object_insts(vid)
    mean_ap, ap_class = eval_video_object(groundtruth, prediction)
    return mean_ap, ap_class


def evaluate_relation(dataset, split, prediction):
    groundtruth = dict()
    for vid in dataset.get_index(split):
        groundtruth[vid] = dataset.get_relation_insts(vid)
    mean_ap, rec_at_n, mprec_at_n = eval_video_relation(groundtruth, prediction)
    # evaluate in zero-shot setting
    print('[info] zero-shot setting')
    zeroshot_triplets = dataset.get_triplets(split).difference(
            dataset.get_triplets('train'))
    groundtruth = dict()
    zs_prediction = dict()
    for vid in dataset.get_index(split):
        gt_relations = dataset.get_relation_insts(vid)
        zs_gt_relations = []
        for r in gt_relations:
            if tuple(r['triplet']) in zeroshot_triplets:
                zs_gt_relations.append(r)
        if len(zs_gt_relations) > 0:
            groundtruth[vid] = zs_gt_relations
            zs_prediction[vid] = prediction.get(vid, [])
    zs_mean_ap, zs_rec_at_n, zs_mprec_at_n = eval_video_relation(groundtruth, zs_prediction)
    return mean_ap, rec_at_n, mprec_at_n, zs_mean_ap, zs_rec_at_n, zs_mprec_at_n


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate a set of tasks related to video relation understanding.')
    parser.add_argument('dataset', choices=['imagenet-vidvrd', 'vidor'], help='the dataset name for evaluation')
    parser.add_argument('split', choices=['validation', 'testing'], help='the split name for evaluation')
    parser.add_argument('task', choices=['object', 'relation'], help='which task to evaluate')
    parser.add_argument('prediction', type=str, help='Corresponding prediction JSON file')
    args = parser.parse_args()

    if args.dataset=='imagenet-vidvrd':
        if args.split=='testing':
            args.split = 'test'
        else:
            print('[warning] there is no validation set in ImageNet-VidVRD dataset')
        if args.task=='relation':
            # load train set for zero-shot evaluation
            dataset = ImagenetVidVRD('../vidvrd-dataset', '../vidvrd-dataset/videos', ['train', args.split])
        else:
            dataset = ImagenetVidVRD('../vidvrd-dataset', '../vidvrd-dataset/videos', [args.split])
    elif args.dataset=='vidor':
        if args.task=='relation':
            # load train set for zero-shot evaluation
            dataset = VidOR('../vidor-dataset/annotation', '../vidor-dataset/video', ['training', args.split], low_memory=True)
        else:
            dataset = VidOR('../vidor-dataset/annotation', '../vidor-dataset/video', [args.split], low_memory=True)
    else:
        raise Exception('Unknown dataset {}'.format(args.dataset))

    print('Loading prediction from {}'.format(args.prediction))
    with lzma.open(args.prediction, 'rt') as fin:
        pred = json.load(fin)
    print('Number of videos in prediction: {}'.format(len(pred['results'])))

    if args.task=='object':
        evaluate_object(dataset, args.split, pred['results'])
    elif args.task=='relation':
        evaluate_relation(dataset, args.split, pred['results'])
