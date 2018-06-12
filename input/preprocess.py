import os
import sys

import cv2
import numpy as np
import numpy.random as npr
from pycocotools.coco import COCO

sys.path.append(os.curdir)

from config import trainImage_path
from config import pixel_average
from config import random_resize
from config import train_annFile
from config import train_type
from config import anchor_scales
from config import anchor_ratios
from config import feat_strides
from config import posi_anchor_thresh
from config import neg_anchor_thresh
from config import input_shape
from config import batch_anchor_num
from config import positive_ratio

coco = COCO(train_annFile)

from lib import bbox_overlaps
from lib import draw_boxes


def data_augmentation(image, raw_size, bboxs, categories, segmentations):
    """

    :param image: 输入图片
    :param raw_size: (高，宽)输入图片的大小
    :param bboxs: [N, (x1, y1, x2, y2)]
    :param segmentations:  一个长度为N的列表，列表的每个元素是一个[M, 2]数组，每行是一个坐标，
    :param categories： [N],所属的类别
    :return: crop_and_resize_image, bboxs, segmentations
    """
    # randomly choose a size and resize the image to that size
    image -= pixel_average  # 去均值
    rdm_ratio = random_resize[npr.randint(0, len(random_resize))]  # 0.5， 1， 1.5， 2

    new_size = raw_size * rdm_ratio
    image = cv2.resize(image, tuple((int(new_size[1]), int(new_size[0]))))

    bboxs = np.asarray(np.array(bboxs, dtype=np.float32) * rdm_ratio, dtype=np.int32)
    segmentations = [seg * rdm_ratio for seg in segmentations]

    # crop some area form image
    image, bboxs, segmentations, categories = crop_area(image, bboxs, segmentations, categories)

    # resize the cropped image to input; size ratio = raw size / input size
    resize_ratio = [image.shape[x] / input_shape[x] for x in range(2)]
    image = cv2.resize(image, input_shape)
    bboxs[:, [0, 2]] = bboxs[:, [0, 2]] / resize_ratio[1]
    bboxs[:, [1, 3]] = bboxs[:, [1, 3]] / resize_ratio[0]

    for idx in range(len(segmentations)):
        segmentations[idx][:, 0] = segmentations[idx][:, 0] / resize_ratio[1]
        segmentations[idx][:, 1] = segmentations[idx][:, 1] / resize_ratio[0]

    image_draw = draw_boxes(image.copy(), bboxs)
    cv2.imwrite('./demo.jpg', image_draw)

    return image, bboxs, categories, segmentations

# image, bboxs, segmentations, categories
def crop_area(im, bboxs, polys, tags, crop_background=False, max_tries=50):
    '''
    make random crop from the input image
    :param im: [高，宽，通道数] 已经去均值了
    :param bboxs：[N, (x1, y1, x2, y2)]
    :param polys: 一个长度为N的列表，列表的每个元素是一个[M, 2]数组，每行是一个坐标，
    :param tags: 标签
    :param crop_background: 是否裁剪背景
    :param max_tries:
    :return:
    '''

    h, w, c = im.shape
    pad_h = h // 10
    pad_w = w // 10

    h_array = np.zeros((h + pad_h*2,), dtype=np.int32)
    w_array = np.zeros((w + pad_w*2,), dtype=np.int32)
    for box in bboxs:
        w_array[box[0] + pad_w:box[2] + pad_w] = 1
        h_array[box[1] + pad_h:box[3] + pad_h] = 1

    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    # if the there is nowhere to crop
    if len(h_axis) == 0 or len(w_axis) == 0:
        return im, bboxs, polys, tags
    for i in range(max_tries):
        xx = np.random.choice(w_axis, size=2)
        yy = np.random.choice(h_axis, size=2)

        xmin = np.min(xx) - pad_w
        xmax = np.max(xx) - pad_w
        xmin = np.clip(xmin, 0, w - 1)
        xmax = np.clip(xmax, 0, w - 1)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_h
        ymin = np.clip(ymin, 0, h - 1)
        ymax = np.clip(ymax, 0, h - 1)

        if xmax - xmin < 0.3 * w or ymax - ymin < 0.3 * h:
            continue
        if len(bboxs) != 0:
            # 找出在裁剪区域以内的框框
            poly_axis_in_area = (bboxs[:, 0] >= xmin) & (bboxs[:, 2] <= xmax) \
                                & (bboxs[:, 1] >= ymin) & (bboxs[:, 3] <= ymax)

            selected_polys = np.where(poly_axis_in_area)[0]
        else:
            selected_polys = []

        im = im[ymin:ymax + 1, xmin:xmax + 1, :]
        polys = [polys[x] for x in selected_polys]
        tags = [tags[x] for x in selected_polys]
        bboxs = bboxs[selected_polys]

        if len(selected_polys) == 0:
            if crop_background:
                return im, bboxs, polys, tags
            else:
                continue
        for idx, poly in enumerate(polys):
            polys[idx][0] -= xmin
            polys[idx][1] -= ymin

        bboxs[:, 0] -= xmin
        bboxs[:, 1] -= ymin
        bboxs[:, 2] -= xmin
        bboxs[:, 3] -= ymin
        return im, bboxs, polys, tags

    return im, bboxs, polys, tags


def cls_target(img_shape, bboxes):
    scales = np.array(anchor_scales).reshape((-1, 1))
    ratios = np.array(anchor_ratios)

    all_scales = (scales * ratios).reshape(-1)

    labels = []
    targets = []
    anchors = []

    num_anchors = len(all_scales)
    for feat_stride in feat_strides:
        per_cell_anchor = np.zeros([num_anchors, 4], dtype=np.float32)
        per_cell_anchor[:, 0] = (feat_stride - 1) / 2 - all_scales / 2  # xmin
        per_cell_anchor[:, 2] = (feat_stride - 1) / 2 + all_scales / 2  # xmax
        per_cell_anchor[:, 1] = (feat_stride - 1) / 2 - all_scales / 2  # ymin
        per_cell_anchor[:, 3] = (feat_stride - 1) / 2 + all_scales / 2  # ymax

        fm_h = img_shape[0] // feat_stride
        fm_w = img_shape[1] // feat_stride

        # every predict feature map pixel has num_scales anchors,
        # each anchor has a label, as well as the target
        label = np.empty((fm_h * fm_w * num_anchors,), dtype=np.int8)
        label.fill(-1)
        target = np.empty((fm_h * fm_w * num_anchors, 4), dtype=np.float32)

        shift_x = np.arange(0, fm_w) * feat_stride
        shift_y = np.arange(0, fm_h) * feat_stride
        shift_x, shift_y = np.meshgrid(shift_x, shift_y)
        #
        shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                            shift_x.ravel(), shift_y.ravel())).transpose()
        all_anchors = (per_cell_anchor.reshape((1, num_anchors, 4)) +
                       shifts.reshape((1, fm_h * fm_w, 4)).transpose((1, 0, 2)))

        all_anchors = all_anchors.reshape((fm_h * fm_w * num_anchors, 4))

        overlaps = bbox_overlaps(
            np.ascontiguousarray(all_anchors, dtype=np.float),
            np.ascontiguousarray(bboxes, dtype=np.float))

        argmax_overlaps = overlaps.argmax(axis=1)

        # print(argmax_overlaps.shape)
        max_overlaps = overlaps[np.arange(all_anchors.shape[0]), argmax_overlaps]
        # print(max_overlaps)

        all_pos_idx = np.where(max_overlaps > posi_anchor_thresh)[0]

        all_neg_idx = np.where((max_overlaps < neg_anchor_thresh) & (max_overlaps > 0))[0]

        if batch_anchor_num * positive_ratio > len(all_pos_idx):
            posi_idx = all_pos_idx

        else:
            posi_idx = npr.choice(all_pos_idx, int(batch_anchor_num * positive_ratio))

        neg_idx = npr.choice(all_neg_idx, int(len(posi_idx) * (1 - positive_ratio) / positive_ratio))

        # TODO some feature map have no posi anchors
        label[posi_idx] = 1
        label[neg_idx] = 0
        if len(neg_idx) == 0:
            neg_idx = npr.choice(all_neg_idx, int(batch_anchor_num * (1 - positive_ratio)))
            label[neg_idx] = 0

        box_target = np.zeros((all_anchors.shape[0], 4), dtype=np.float64)
        # pos_idx 存放的是要训练的正例的　在all_anchor中的索引
        # argmax_overlaps 存放的是每个anchor与所有gt_text交集中最大的gt在text_proposal_gt中的索引, 长度是所有anchor的个数
        # text_proposal_gt　存放所有的gt_text_proposal
        posi_target = bboxes[argmax_overlaps[posi_idx]]
        box_target[posi_idx] = posi_target

        # print(posi_idx)

        labels.append(label)
        targets.append(target)
        anchors.append(all_anchors)
    return labels, targets, anchors


def mask_target(img_shape, segmentations):
    h, w = img_shape[:2]

    segmentation_mask = np.zeros((h, w), dtype=np.int32)

    for seg in segmentations:
        seg = np.array(seg, dtype=np.int32)

        cv2.fillPoly(segmentation_mask, [seg], 1)

    return segmentation_mask

