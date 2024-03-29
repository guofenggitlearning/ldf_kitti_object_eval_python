import numba
import numpy as np

from rotate_iou import rotate_iou_gpu_eval


def get_split_parts(num_sample, num_part):
    """

    Args:
        num_sample: int, the number of total samples
        num_part: int, a parameter for fast calculate algorithm

    Returns:
        split_parts: list of int

    """
    same_part = num_sample // num_part
    remain_num = num_sample % num_part
    if same_part == 0:
        return [num_sample]

    if remain_num == 0:
        return [same_part] * num_part
    else:
        return [same_part] * num_part + [remain_num]


@numba.jit
def get_thresholds(scores: np.ndarray, num_gt, num_sample_pt=41):
    """

    Args:
        scores: ndarray of float, [num_tp], scores of true positive detections
        num_gt: int, the number of valid ground truth objects
        num_sample_pt: int

    Returns:
        thresholds: list of float, sampled scores, the maximum length of which is num_sample_pt

    """
    scores.sort()
    scores = scores[::-1]
    current_recall = 0
    thresholds = []
    for i, score in enumerate(scores):
        l_recall = (i + 1) / num_gt
        if i < (len(scores) - 1):
            r_recall = (i + 2) / num_gt
        else:
            r_recall = l_recall
        if r_recall - current_recall < current_recall - l_recall and i < len(scores) - 1:
            continue
        thresholds.append(score)
        current_recall += 1 / (num_sample_pt - 1.0)
    return thresholds


@numba.jit(nopython=True)
def image_box_overlap(boxes, query_boxes, criterion=-1):
    """

    Args:
        boxes: ndarray of float, [N, 4], xyxy format
        query_boxes: ndarray of float, [K, 4], xyxy format
        criterion: the calculation type of the union area

    Returns:
        overlaps: ndarray of float, [N, K]

    """
    N = boxes.shape[0]
    K = query_boxes.shape[0]
    overlaps = np.zeros((N, K), dtype=boxes.dtype)
    for k in range(K):
        qbox_area = ((query_boxes[k, 2] - query_boxes[k, 0]) *
                     (query_boxes[k, 3] - query_boxes[k, 1]))
        for n in range(N):
            iw = (min(boxes[n, 2], query_boxes[k, 2]) -
                  max(boxes[n, 0], query_boxes[k, 0]))
            if iw > 0:
                ih = (min(boxes[n, 3], query_boxes[k, 3]) -
                      max(boxes[n, 1], query_boxes[k, 1]))
                if ih > 0:
                    if criterion == -1:
                        ua = (boxes[n, 2] - boxes[n, 0]) * (boxes[n, 3] - boxes[n, 1]) + qbox_area - iw * ih
                    elif criterion == 0:
                        ua = (boxes[n, 2] - boxes[n, 0]) * (boxes[n, 3] - boxes[n, 1])
                    elif criterion == 1:
                        ua = qbox_area
                    else:
                        ua = 1.0
                    overlaps[n, k] = iw * ih / ua
    return overlaps


def bev_box_overlap(boxes, qboxes, criterion=-1):
    """

    Args:
        boxes: ndarray of float, [N, 5], centers, dims, angles (clockwise when positive)
        qboxes: ndarray of float, [K, 5], centers, dims, angles (clockwise when positive)
        criterion:

    Returns:
        riou: ndarray of float, [N, K]

    """
    riou = rotate_iou_gpu_eval(boxes, qboxes, criterion)
    return riou


@numba.jit(nopython=True, parallel=True)
def d3_box_overlap_kernel(boxes, qboxes, rinc, criterion=-1):
    # only support overlap in the camera coordinates, not the lider coordinates.
    N = boxes.shape[0]
    K = qboxes.shape[0]
    for i in range(N):
        for j in range(K):
            if rinc[i, j] > 0:
                iw = (min(boxes[i, 1], qboxes[j, 1]) - max(
                    boxes[i, 1] - boxes[i, 4], qboxes[j, 1] - qboxes[j, 4]))

                if iw > 0:
                    area1 = boxes[i, 3] * boxes[i, 4] * boxes[i, 5]
                    area2 = qboxes[j, 3] * qboxes[j, 4] * qboxes[j, 5]
                    inc = iw * rinc[i, j]
                    if criterion == -1:
                        ua = (area1 + area2 - inc)
                    elif criterion == 0:
                        ua = area1
                    elif criterion == 1:
                        ua = area2
                    else:
                        ua = inc
                    rinc[i, j] = inc / ua
                else:
                    rinc[i, j] = 0.0


def d3_box_overlap(boxes, qboxes, criterion=-1):
    """

    Args:
        boxes: ndarray of float, [N, 7], centers, dims, angles
        qboxes: ndarray of float, [K, 7], centers, dims, angles
        criterion:

    Returns:
        rinc: ndarray of float, [N, K]

    """
    rinc = rotate_iou_gpu_eval(boxes[:, [0, 2, 3, 5, 6]],
                               qboxes[:, [0, 2, 3, 5, 6]], 2)
    d3_box_overlap_kernel(boxes, qboxes, rinc, criterion)
    return rinc


@numba.jit(nopython=True)
def compute_statistics_jit(overlaps, gt_datas, dt_datas, ignored_gt, ignored_dt, dc_bboxes,
                           metric, min_overlap, score_thresh=0.0, compute_fp=False, compute_aos=False):
    """

    Args:
        overlaps: ndarray of float, [num_gt, num_dt]
        gt_datas: ndarray of float, [num_gt, 5], bboxes, alphas
        dt_datas: ndarray of float, [num_dt, 6], bboxes, alphas, scores
        ignored_gt: ndarray of int, [num_gt], 0: not ignored, 1: ignored, -1: unknown
        ignored_dt: ndarray of int, [num_dt], 0: not ignored, 1: ignored, -1: unknown
        dc_bboxes: ndarray of float, [num_dc, 4]
        metric: int, the evaluation type, 0: bbox, 1: bev, 2: 3d
        min_overlap: float
        score_thresh: float
        compute_fp: bool
        compute_aos: bool

    Returns:
        tp: int, the number of true positive detections
        fp: int, the number of false positive detections
        fn: int, the number of false negative detections
        similarity: float
        thresholds: ndarray of float, [num_tp], scores of true positive detections

    """
    gt_size = gt_datas.shape[0]
    dt_size = dt_datas.shape[0]

    gt_alphas = gt_datas[:, 4]
    dt_alphas = dt_datas[:, 4]
    dt_bboxes = dt_datas[:, :4]
    dt_scores = dt_datas[:, -1]

    assigned_detection = [False] * dt_size
    ignored_threshold = [False] * dt_size
    if compute_fp:
        for i in range(dt_size):
            if dt_scores[i] < score_thresh:
                ignored_threshold[i] = True
    NO_DETECTION = -10000000
    tp, fp, fn, similarity = 0, 0, 0, 0
    thresholds = np.zeros((gt_size, ))
    thresh_idx = 0
    delta = np.zeros((gt_size, ))
    delta_idx = 0
    for i in range(gt_size):
        if ignored_gt[i] == -1:
            continue
        det_idx = -1
        valid_detection = NO_DETECTION
        max_overlap = 0
        assigned_ignored_dt = False

        for j in range(dt_size):
            if ignored_dt[j] == -1:
                continue
            if assigned_detection[j]:
                continue
            if ignored_threshold[j]:
                continue
            overlap = overlaps[i, j]
            dt_score = dt_scores[j]
            if not compute_fp and overlap > min_overlap and dt_score > valid_detection:
                det_idx = j
                valid_detection = dt_score
            elif compute_fp and overlap > min_overlap and (overlap > max_overlap or assigned_ignored_dt) and \
                    ignored_dt[j] == 0:
                max_overlap = overlap
                det_idx = j
                valid_detection = 1
                assigned_ignored_dt = False
            elif compute_fp and overlap > min_overlap and valid_detection == NO_DETECTION and \
                    ignored_dt[j] == 1:
                det_idx = j
                valid_detection = 1
                assigned_ignored_dt = True

        if valid_detection == NO_DETECTION and ignored_gt[i] == 0:
            fn += 1
        elif valid_detection != NO_DETECTION and (ignored_gt[i] == 1 or ignored_dt[det_idx] == 1):
            assigned_detection[det_idx] = True
        elif valid_detection != NO_DETECTION:
            tp += 1
            thresholds[thresh_idx] = dt_scores[det_idx]
            thresh_idx += 1
            if compute_aos:
                delta[delta_idx] = gt_alphas[i] - dt_alphas[det_idx]
                delta_idx += 1
            assigned_detection[det_idx] = True
    if compute_fp:
        for i in range(dt_size):
            if not (assigned_detection[i] or ignored_dt[i] == -1 or ignored_dt[i] == 1 or ignored_threshold[i]):
                fp += 1
        nstuff = 0
        if metric == 0:
            overlaps_dt_dc = image_box_overlap(dt_bboxes, dc_bboxes, 0)
            for i in range(dt_size):
                for j in range(dc_bboxes.shape[0]):
                    if assigned_detection[i]:
                        continue
                    if ignored_dt[i] == -1 or ignored_dt[i] == 1:
                        continue
                    if ignored_threshold[i]:
                        continue
                    if overlaps_dt_dc[i, j] > min_overlap:
                        assigned_detection[i] = True
                        nstuff += 1
        fp -= nstuff
        if compute_aos:
            tmp = np.zeros((fp + delta_idx, ))
            for i in range(delta_idx):
                tmp[i + fp] = (1.0 + np.cos(delta[i])) / 2.0
            if tp > 0 or fp > 0:
                similarity = np.sum(tmp)
            else:
                similarity = -1
    return tp, fp, fn, similarity, thresholds[:thresh_idx]


@numba.jit(nopython=True)
def fused_compute_statistics(overlaps, pr, gt_nums, dt_nums, dc_nums, gt_datas, dt_datas, dontcares,
                             ignored_gts, ignored_dts, metric, min_overlap, thresholds, compute_aos=False):
    """

    Args:
        overlaps: ndarray of float, [num_gt_per_part, num_dt_per_part]
        pr: ndarray of int, [about 41, 4], all zeros
        gt_nums: ndarray of int, [parted_num]
        dt_nums: ndarray of int, [parted_num]
        dc_nums: ndarray of int, [parted_num]
        gt_datas: ndarray of float, [num_gt_per_part, 5], bboxes, alphas
        dt_datas: ndarray of float, [num_dt_per_part, 6], bboxes, alphas, scores
        dontcares: ndarray of float, [num_dc_per_part, 4]
        ignored_gts: ndarray of int, [num_gt_per_part], 0: not ignored, 1: ignored, -1: unknown
        ignored_dts: ndarray of int, [num_gt_per_part], 0: not ignored, 1: ignored, -1: unknown
        metric: int, the evaluation type, 0: bbox, 1: bev, 2: 3d
        min_overlap: float
        thresholds: ndarray of float, [about 41], about 41 scores of true positive detections
        compute_aos: bool

    Returns:

    """
    gt_num = 0
    dt_num = 0
    dc_num = 0
    for i in range(gt_nums.shape[0]):
        for t, score_thresh in enumerate(thresholds):
            overlap = overlaps[gt_num:gt_num + gt_nums[i], dt_num:dt_num + dt_nums[i]]
            gt_data = gt_datas[gt_num:gt_num + gt_nums[i]]
            dt_data = dt_datas[dt_num:dt_num + dt_nums[i]]
            ignored_gt = ignored_gts[gt_num:gt_num + gt_nums[i]]
            ignored_dt = ignored_dts[dt_num:dt_num + dt_nums[i]]
            dontcare = dontcares[dc_num:dc_num + dc_nums[i]]
            tp, fp, fn, similarity, _ = compute_statistics_jit(
                overlap,
                gt_data,
                dt_data,
                ignored_gt,
                ignored_dt,
                dontcare,
                metric,
                min_overlap=min_overlap,
                score_thresh=score_thresh,
                compute_fp=True,
                compute_aos=compute_aos)
            pr[t, 0] += tp
            pr[t, 1] += fp
            pr[t, 2] += fn
            if similarity != -1:
                pr[t, 3] += similarity
        gt_num += gt_nums[i]
        dt_num += dt_nums[i]
        dc_num += dc_nums[i]


def calculate_iou_partly(gt_annos, dt_annos, metric, num_part=50):
    """
    this function can calculate iou in bbox, bev and 3d, determined by the parameter 'metric',
    and must be used in the camera coordinates.

    Args:
        gt_annos: list of dict, must from get_label_annos() in kitti_common.py
        dt_annos: list of dict, must from get_label_annos() in kitti_common.py
        metric: int, the evaluation type, 0: bbox, 1: bev, 2: 3d
        num_part: int, a parameter for fast calculate algorithm

    Returns:
        overlaps: list of ndarray of float, [[num_gt_per_sample, num_dt_per_sample], ...], the length is num_sample
        parted_overlaps: list of ndarray of float, [[num_gt_per_part, num_dt_per_part], ...], the length is num_part
        total_gt_num: ndarray of int, [num_example], the number of ground truth objects
        total_dt_num: ndarray of int, [num_example], the number of detected objects

    """
    assert len(gt_annos) == len(dt_annos)
    num_example = len(gt_annos)
    split_parts = get_split_parts(num_example, num_part)

    total_gt_num = np.stack([len(a["name"]) for a in gt_annos], 0)  # [num_example]
    total_dt_num = np.stack([len(a["name"]) for a in dt_annos], 0)  # [num_example]
    parted_overlaps = []
    example_idx = 0

    for parted_num in split_parts:
        gt_annos_part = gt_annos[example_idx:example_idx + parted_num]
        dt_annos_part = dt_annos[example_idx:example_idx + parted_num]
        if metric == 0:
            gt_boxes = np.concatenate([a["bbox"] for a in gt_annos_part], 0)  # [N, 4]
            dt_boxes = np.concatenate([a["bbox"] for a in dt_annos_part], 0)  # [K, 4]
            overlap_part = image_box_overlap(gt_boxes, dt_boxes)  # [N, K]
        elif metric == 1:
            loc = np.concatenate([a["location"][:, [0, 2]] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"][:, [0, 2]] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a["location"][:, [0, 2]] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"][:, [0, 2]] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = bev_box_overlap(gt_boxes, dt_boxes).astype(np.float64)  # [N, K]
        elif metric == 2:
            loc = np.concatenate([a["location"] for a in gt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in gt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in gt_annos_part], 0)
            gt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            loc = np.concatenate([a["location"] for a in dt_annos_part], 0)
            dims = np.concatenate([a["dimensions"] for a in dt_annos_part], 0)
            rots = np.concatenate([a["rotation_y"] for a in dt_annos_part], 0)
            dt_boxes = np.concatenate(
                [loc, dims, rots[..., np.newaxis]], axis=1)
            overlap_part = d3_box_overlap(gt_boxes, dt_boxes).astype(np.float64)  # [N, K]
        else:
            raise ValueError("unknown metric")
        parted_overlaps.append(overlap_part)
        example_idx += parted_num
    overlaps = []
    example_idx = 0
    for j, parted_num in enumerate(split_parts):
        gt_num_idx, dt_num_idx = 0, 0
        for i in range(parted_num):
            gt_box_num = total_gt_num[example_idx + i]
            dt_box_num = total_dt_num[example_idx + i]
            overlaps.append(
                # [num_gt_per_sample, num_dt_per_sample]
                parted_overlaps[j][gt_num_idx:gt_num_idx + gt_box_num, dt_num_idx:dt_num_idx + dt_box_num]
            )
            gt_num_idx += gt_box_num
            dt_num_idx += dt_box_num
        example_idx += parted_num

    return overlaps, parted_overlaps, total_gt_num, total_dt_num


def clean_data(gt_anno, dt_anno, current_class, difficulty):
    """

    Args:
        gt_anno: dict, annotations per sample
        dt_anno: dict, detected results per sample
        current_class: int
        difficulty: int

    Returns:
        num_valid_gt: int, the number of valid ground truth objects per sample
        ignored_gt: list of int, the length is num_gt, 0: not ignored, 1: ignored, -1: unknown
        ignored_dt: list of int, the length is num_dt, 0: not ignored, 1: ignored, -1: unknown
        dc_bboxes: list of ndarray of float, [[4], ...], DontCare bboxes in annotations

    """
    CLASS_NAMES = ['car', 'pedestrian', 'cyclist', 'van', 'person_sitting', 'truck']

    # MIN_HEIGHT = [40, 25, 25]
    # MAX_OCCLUSION = [0, 1, 2]
    # MAX_TRUNCATION = [0.15, 0.3, 0.5]

    MIN_DISTANCE = [0, 10, 20, 30, 40, 50, 60, 70]
    MAX_DISTANCE = [10, 20, 30, 40, 50, 60, 70, 80]

    dc_bboxes, ignored_gt, ignored_dt = [], [], []
    current_cls_name = CLASS_NAMES[current_class].lower()
    num_gt = len(gt_anno["name"])
    num_dt = len(dt_anno["name"])
    num_valid_gt = 0
    for i in range(num_gt):
        gt_name = gt_anno["name"][i].lower()
        if gt_name == current_cls_name:
            valid_class = 1
        elif gt_name == "Person_sitting".lower() and current_cls_name == "Pedestrian".lower():
            valid_class = 0
        elif gt_name == "Van".lower() and current_cls_name == "Car".lower():
            valid_class = 0
        else:
            valid_class = -1

        # ignore = False
        # height = gt_anno["bbox"][i, 3] - gt_anno["bbox"][i, 1]
        # if gt_anno["occluded"][i] > MAX_OCCLUSION[difficulty] or \
        #         gt_anno["truncated"][i] > MAX_TRUNCATION[difficulty] or \
        #         height <= MIN_HEIGHT[difficulty]:
        #     ignore = True

        ignore = False
        cur_diff = -1
        distance = (gt_anno["location"][i, 0] ** 2 + gt_anno["location"][i, 2] ** 2) ** 0.5
        for l in range(len(MAX_DISTANCE)):
            if distance >= MIN_DISTANCE[l] and distance < MAX_DISTANCE[l]:
                cur_diff = l
                break
        if cur_diff != difficulty:
            ignore = True

        if valid_class == 1 and not ignore:
            ignored_gt.append(0)
            num_valid_gt += 1
        elif valid_class == 0 or (ignore and valid_class == 1):
            ignored_gt.append(1)
        else:
            ignored_gt.append(-1)
        if gt_anno["name"][i] == "DontCare":
            dc_bboxes.append(gt_anno["bbox"][i])

    for i in range(num_dt):
        if dt_anno["name"][i].lower() == current_cls_name:
            valid_class = 1
        else:
            valid_class = -1

        # height = abs(dt_anno["bbox"][i, 3] - dt_anno["bbox"][i, 1])
        # if height < MIN_HEIGHT[difficulty]:
        #     ignored_dt.append(1)
        # elif valid_class == 1:
        #     ignored_dt.append(0)
        # else:
        #     ignored_dt.append(-1)

        if valid_class == 1:
            ignored_dt.append(0)
        else:
            ignored_dt.append(-1)

    return num_valid_gt, ignored_gt, ignored_dt, dc_bboxes


def _prepare_data(gt_annos, dt_annos, current_class, difficulty):
    """

    Args:
        gt_annos: list of dict, must from get_label_annos() in kitti_common.py
        dt_annos: list of dict, must from get_label_annos() in kitti_common.py
        current_class: int, 0: car, 1: pedestrian, 2: cyclist
        difficulty: int, the evaluation difficulty, 0: easy, 1: normal, 2: hard

    Returns:
        gt_datas_list: list of ndarray of float, [[num_gt_per_sample, 5], ...], bboxes, alphas
        dt_datas_list: list of ndarray of float, [[num_dt_per_sample, 6], ...], bboxes, alphas, scores
        ignored_gts: list of ndarray of int, [[num_gt_per_sample], ...], 0: not ignored, 1: ignored, -1: unknown
        ignored_dts: list of ndarray of int, [[num_dt_per_sample], ...], 0: not ignored, 1: ignored, -1: unknown
        dontcares: list of ndarray of float, [[num_dc, 4], ...]
        total_dc_num: ndarray of int, [num_sample], each of which is the number of DontCare bboxes per sample
        total_num_valid_gt: int, the number of valid ground truth objects in all samples

    """
    gt_datas_list = []
    dt_datas_list = []
    total_dc_num = []
    ignored_gts, ignored_dts, dontcares = [], [], []
    total_num_valid_gt = 0
    for i in range(len(gt_annos)):
        # num_valid_gt: int, the number of valid ground truth objects per sample
        # ignored_gt: list of int, the length is num_gt, 0: not ignored, 1: ignored, -1: unknown
        # ignored_dt: list of int, the length is num_dt, 0: not ignored, 1: ignored, -1: unknown
        # dc_bboxes: list of ndarray of float, [[4], ...], DontCare bboxes in annotations
        rets = clean_data(gt_annos[i], dt_annos[i], current_class, difficulty)
        num_valid_gt, ignored_gt, ignored_dt, dc_bboxes = rets

        ignored_gts.append(np.array(ignored_gt, dtype=np.int64))
        ignored_dts.append(np.array(ignored_dt, dtype=np.int64))
        if len(dc_bboxes) == 0:
            dc_bboxes = np.zeros((0, 4)).astype(np.float64)
        else:
            dc_bboxes = np.stack(dc_bboxes, 0).astype(np.float64)
        total_dc_num.append(dc_bboxes.shape[0])
        dontcares.append(dc_bboxes)
        total_num_valid_gt += num_valid_gt
        gt_datas = np.concatenate([
            gt_annos[i]["bbox"], gt_annos[i]["alpha"][..., np.newaxis]
        ], 1)
        dt_datas = np.concatenate([
            dt_annos[i]["bbox"], dt_annos[i]["alpha"][..., np.newaxis], dt_annos[i]["score"][..., np.newaxis]
        ], 1)
        gt_datas_list.append(gt_datas)
        dt_datas_list.append(dt_datas)
    total_dc_num = np.stack(total_dc_num, axis=0)
    return gt_datas_list, dt_datas_list, ignored_gts, ignored_dts, dontcares, total_dc_num, total_num_valid_gt


def eval_class(gt_annos, dt_annos, current_classes, difficultys, metric, min_overlaps,
               compute_aos=False, num_part=100):
    """

    Args:
        gt_annos: list of dict, must from get_label_annos() in kitti_common.py
        dt_annos: list of dict, must from get_label_annos() in kitti_common.py
        current_classes: list of int, 0: car, 1: pedestrian, 2: cyclist
        difficultys: list of int, the evaluation difficulty, 0: easy, 1: normal, 2: hard
        metric: int, the evaluation type, 0: bbox, 1: bev, 2: 3d
        min_overlaps: ndarray of float, [num_minoverlap, num_metric, num_class]
        compute_aos: bool
        num_part: int, a parameter for fast calculate algorithm

    Returns:
        ret: dict,
            'recall': ndarray of float, [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS]
            'precision': ndarray of float, [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS]
            'orientation': ndarray of float, [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS]

    """
    assert len(gt_annos) == len(dt_annos)
    num_example = len(gt_annos)
    split_parts = get_split_parts(num_example, num_part)

    # overlaps: list of ndarray of float, [[num_gt_per_sample, num_dt_per_sample], ...], the length is num_sample
    # parted_overlaps: list of ndarray of float, [[num_gt_per_part, num_dt_per_part], ...], the length is num_part
    # total_gt_num: ndarray of int, [num_example]
    # total_dt_num: ndarray of int, [num_example]
    rets = calculate_iou_partly(gt_annos, dt_annos, metric, num_part)
    overlaps, parted_overlaps, total_gt_num, total_dt_num = rets

    N_SAMPLE_PTS = 41
    num_minoverlap = len(min_overlaps)
    num_class = len(current_classes)
    num_difficulty = len(difficultys)

    precision = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    recall = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])
    aos = np.zeros([num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS])

    for m, current_class in enumerate(current_classes):
        for l, difficulty in enumerate(difficultys):
            # gt_datas_list: list of ndarray of float, [[num_gt_per_sample, 5], ...], bboxes, alphas
            # dt_datas_list: list of ndarray of float, [[num_dt_per_sample, 6], ...], bboxes, alphas, scores
            # ignored_gts: list of ndarray of int, [[num_gt_per_sample], ...], 0: not ignored, 1: ignored, -1: unknown
            # ignored_dts: list of ndarray of int, [[num_dt_per_sample], ...], 0: not ignored, 1: ignored, -1: unknown
            # dontcares: list of ndarray of float, [[num_dc, 4], ...]
            # total_dc_num: ndarray of int, [num_sample], each of which is the number of DontCare bboxes per sample
            # total_num_valid_gt: int, the number of valid ground truth objects in all samples
            rets = _prepare_data(gt_annos, dt_annos, current_class, difficulty)
            gt_datas_list, dt_datas_list, ignored_gts, ignored_dts, dontcares, total_dc_num, total_num_valid_gt = rets

            if metric == 0:
                print('Valid ground truth objects of Class {:d} in Difficulty {:d}: {:d}'.format(
                    current_class, difficulty, total_num_valid_gt))

            for k, min_overlap in enumerate(min_overlaps[:, metric, m]):
                all_thresholds = []
                for i in range(len(gt_annos)):
                    # tp: int, the number of true positive detections
                    # fp: int, the number of false positive detections
                    # fn: int, the number of false negative detections
                    # similarity: float
                    # thresholds: ndarray of float, [num_tp], scores of true positive detections
                    rets = compute_statistics_jit(
                        overlaps[i],  # [num_gt_per_sample, num_dt_per_sample]
                        gt_datas_list[i],  # [num_gt_per_sample, 5]
                        dt_datas_list[i],  # [num_dt_per_sample, 6]
                        ignored_gts[i],  # [num_gt_per_sample]
                        ignored_dts[i],  # [num_dt_per_sample]
                        dontcares[i],  # [num_dc, 4]
                        metric,  # int
                        min_overlap=min_overlap,  # float
                        score_thresh=0.0,  # float
                        compute_fp=False,  # bool
                    )
                    tp, fp, fn, similarity, thresholds = rets
                    all_thresholds += thresholds.tolist()
                all_thresholds = np.array(all_thresholds)

                # thresholds: list of float, sampled scores, the maximum length of which is 41
                thresholds = get_thresholds(all_thresholds, total_num_valid_gt)
                thresholds = np.array(thresholds)

                # pr: ndarray of float, [about 41, 4], tp, fp, fn, similarity
                pr = np.zeros([len(thresholds), 4])
                idx = 0
                for j, parted_num in enumerate(split_parts):
                    gt_datas_part = np.concatenate(gt_datas_list[idx:idx + parted_num], 0)
                    dt_datas_part = np.concatenate(dt_datas_list[idx:idx + parted_num], 0)
                    dc_datas_part = np.concatenate(dontcares[idx:idx + parted_num], 0)
                    ignored_gts_part = np.concatenate(ignored_gts[idx:idx + parted_num], 0)
                    ignored_dts_part = np.concatenate(ignored_dts[idx:idx + parted_num], 0)
                    fused_compute_statistics(
                        parted_overlaps[j],  # [num_gt_per_part, num_dt_per_part]
                        pr,  # [about 41, 4]
                        total_gt_num[idx:idx + parted_num],  # [parted_num]
                        total_dt_num[idx:idx + parted_num],  # [parted_num]
                        total_dc_num[idx:idx + parted_num],  # [parted_num]
                        gt_datas_part,  # [num_gt_per_part, 5]
                        dt_datas_part,  # [num_dt_per_part, 6]
                        dc_datas_part,  # [num_dc_per_part, 4]
                        ignored_gts_part,  # [num_gt_per_part]
                        ignored_dts_part,  # [num_dt_per_part]
                        metric,  # int
                        min_overlap=min_overlap,  # float
                        thresholds=thresholds,  # [about 41]
                        compute_aos=compute_aos,  # bool
                    )
                    idx += parted_num
                for i in range(len(thresholds)):
                    recall[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 2])
                    precision[m, l, k, i] = pr[i, 0] / (pr[i, 0] + pr[i, 1])
                    if compute_aos:
                        aos[m, l, k, i] = pr[i, 3] / (pr[i, 0] + pr[i, 1])
                for i in range(len(thresholds)):
                    recall[m, l, k, i] = np.max(recall[m, l, k, i:], axis=-1)
                    precision[m, l, k, i] = np.max(precision[m, l, k, i:], axis=-1)
                    if compute_aos:
                        aos[m, l, k, i] = np.max(aos[m, l, k, i:], axis=-1)
    ret_dict = {
        "recall": recall,
        "precision": precision,
        "orientation": aos,
    }
    return ret_dict


def get_mAP(prec):
    sums = 0
    for i in range(0, prec.shape[-1], 4):
        sums = sums + prec[..., i]
    return sums / 11 * 100


def get_mAP_R40(prec):
    sums = 0
    for i in range(1, prec.shape[-1]):
        sums = sums + prec[..., i]
    return sums / 40 * 100


def do_eval(gt_annos, dt_annos, current_classes, difficultys, min_overlaps, compute_aos=False):
    """

    Args:
        gt_annos: list of dict, must from get_label_annos() in kitti_common.py
        dt_annos: list of dict, must from get_label_annos() in kitti_common.py
        current_classes: list of int, 0: car, 1: pedestrian, 2: cyclist
        difficultys: list of int, the evaluation difficulty, 0: easy, 1: normal, 2: hard
        min_overlaps: ndarray of float, [num_minoverlap, num_metric, num_class]
        compute_aos: bool

    Returns:
        mAP result: ndarray of float, [num_class, num_difficulty, num_minoverlap]

    """
    # ret['recall']: ndarray of float, [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS]
    # ret['precision']: ndarray of float, [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS]
    # ret['orientation']: ndarray of float, [num_class, num_difficulty, num_minoverlap, N_SAMPLE_PTS]
    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 0, min_overlaps, compute_aos)
    mAP_bbox = get_mAP(ret["precision"])
    mAP_bbox_R40 = get_mAP_R40(ret["precision"])

    mAP_aos = None
    mAP_aos_R40 = None
    if compute_aos:
        mAP_aos = get_mAP(ret["orientation"])
        mAP_aos_R40 = get_mAP_R40(ret["orientation"])

    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 1, min_overlaps)
    mAP_bev = get_mAP(ret["precision"])
    mAP_bev_R40 = get_mAP_R40(ret["precision"])

    ret = eval_class(gt_annos, dt_annos, current_classes, difficultys, 2, min_overlaps)
    mAP_3d = get_mAP(ret["precision"])
    mAP_3d_R40 = get_mAP_R40(ret["precision"])

    return mAP_bbox, mAP_bev, mAP_3d, mAP_aos, mAP_bbox_R40, mAP_bev_R40, mAP_3d_R40, mAP_aos_R40


def get_official_eval_result(gt_annos, dt_annos, current_classes):
    """

    Args:
        gt_annos: list of dict, must from get_label_annos() in kitti_common.py
        dt_annos: list of dict, must from get_label_annos() in kitti_common.py
        current_classes: int or list of int or list of str, desired classes

    Returns:
        result: str

    """
    class_to_name = {
        0: 'Car',
        1: 'Pedestrian',
        2: 'Cyclist',
        3: 'Van',
        4: 'Person_sitting',
        5: 'Truck'
    }
    name_to_class = {v: n for n, v in class_to_name.items()}
    if not isinstance(current_classes, (list, tuple)):
        current_classes = [current_classes]
    current_classes_int = []
    for curcls in current_classes:
        if isinstance(curcls, str):
            current_classes_int.append(name_to_class[curcls])
        else:
            current_classes_int.append(curcls)
    current_classes = current_classes_int

    overlap_0_7 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.7],  # metric 0: bbox
                            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],  # metric 1: bev
                            [0.7, 0.5, 0.5, 0.7, 0.5, 0.7],  # metric 2: 3d
                            ])
    overlap_0_5 = np.array([[0.7, 0.5, 0.5, 0.7, 0.5, 0.5],
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],
                            [0.5, 0.25, 0.25, 0.5, 0.25, 0.5],
                            ])
    # min_overlaps: ndarray of float, [num_minoverlap, num_metric, num_class]
    min_overlaps = np.stack([overlap_0_7, overlap_0_5], axis=0)
    min_overlaps = min_overlaps[:, :, current_classes]

    difficultys = [0, 1, 2, 3, 4, 5, 6, 7]
    num_difficulty = len(difficultys)

    result = ''
    # check whether alpha is valid
    compute_aos = False
    for anno in dt_annos:
        if anno['alpha'].shape[0] != 0:
            if anno['alpha'][0] != -10:
                compute_aos = True
            break

    # mAP result: ndarray of float, [num_class, num_difficulty, num_minoverlap]
    mAPbbox, mAPbev, mAP3d, mAPaos, mAPbbox_R40, mAPbev_R40, mAP3d_R40, mAPaos_R40 = do_eval(
        gt_annos, dt_annos, current_classes, difficultys, min_overlaps, compute_aos)

    for j, curcls in enumerate(current_classes):
        for i in range(min_overlaps.shape[0]):
            result += f'{class_to_name[curcls]} AP:\n'
            result += f'bbox ({min_overlaps[i, 0, j]:.2f}): '
            for l in range(num_difficulty):
                result += f'{mAPbbox[j, l, i]:.4f}'
                result += ', ' if l < num_difficulty - 1 else '\n'
            result += f'bev  ({min_overlaps[i, 1, j]:.2f}): '
            for l in range(num_difficulty):
                result += f'{mAPbev[j, l, i]:.4f}'
                result += ', ' if l < num_difficulty - 1 else '\n'
            result += f'3d   ({min_overlaps[i, 2, j]:.2f}): '
            for l in range(num_difficulty):
                result += f'{mAP3d[j, l, i]:.4f}'
                result += ', ' if l < num_difficulty - 1 else '\n'

            if compute_aos:
                result += f'aos        : '
                for l in range(num_difficulty):
                    result += f'{mAPaos[j, l, i]:.4f}'
                    result += ', ' if l < num_difficulty - 1 else '\n'

            result += f'{class_to_name[curcls]} AP_R40:\n'
            result += f'bbox ({min_overlaps[i, 0, j]:.2f}): '
            for l in range(num_difficulty):
                result += f'{mAPbbox_R40[j, l, i]:.4f}'
                result += ', ' if l < num_difficulty - 1 else '\n'
            result += f'bev  ({min_overlaps[i, 1, j]:.2f}): '
            for l in range(num_difficulty):
                result += f'{mAPbev_R40[j, l, i]:.4f}'
                result += ', ' if l < num_difficulty - 1 else '\n'
            result += f'3d   ({min_overlaps[i, 2, j]:.2f}): '
            for l in range(num_difficulty):
                result += f'{mAP3d_R40[j, l, i]:.4f}'
                result += ', ' if l < num_difficulty - 1 else '\n'

            if compute_aos:
                result += f'aos        : '
                for l in range(num_difficulty):
                    result += f'{mAPaos_R40[j, l, i]:.4f}'
                    result += ', ' if l < num_difficulty - 1 else '\n'

    return result
