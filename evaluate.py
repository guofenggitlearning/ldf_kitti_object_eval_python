import fire

import kitti_common as kitti
from eval import get_official_eval_result


def _read_imageset_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    return [int(line) for line in lines]


def evaluate(result_path,
             label_path='kitti/training/label_2',
             label_split_file='kitti/training/ImageSets/val.txt',
             current_classes=0,
             score_thresh=-1):
    dt_annos = kitti.get_label_annos(result_path)
    if score_thresh > 0:
        dt_annos = kitti.filter_annos_low_score(dt_annos, score_thresh)
    val_image_ids = _read_imageset_file(label_split_file)
    gt_annos = kitti.get_label_annos(label_path, val_image_ids)
    ap_result_str = get_official_eval_result(gt_annos, dt_annos, current_classes)
    print(ap_result_str)


if __name__ == '__main__':
    fire.Fire()
