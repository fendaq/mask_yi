import os
import cv2
import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO
from config import train_annFile
from config import batch_size
from config import trainImage_path
from config import input_shape
from input.preprocess import data_augmentation,cls_target,mask_target

coco = COCO(train_annFile)

def producer():
    # iscrowd: 0 segmentation is polygon
    # iscrowd: 1 segmentation is RLE



    # 以列表的形式返回imageID
    imgIds = coco.getImgIds()

    imgIds_list = tf.constant(imgIds)


    dataset = tf.data.Dataset.from_tensor_slices((imgIds_list,))
    dataset = dataset.map(
        lambda imgId: tuple(tf.py_func(
            #
            sample_handler, [imgId], [tf.float32, tf.int32, tf.int32, tf.int32, tf.int8, tf.float32, tf.float64])),
        num_parallel_calls=1).repeat().batch(batch_size)
    return dataset

def sample_handler(imgId=532481):
    """
    :return: resize_img, labels, target, mask
    """
    print("=====",imgId)
    # 下面这个函数是以列表形式返回的，所以要取[0]
    img_info = coco.loadImgs(int(imgId))[0]
    img_name = img_info['file_name']
    height, width = img_info['height'], img_info['width']

    img_path = os.path.join(trainImage_path, img_name)
    image = cv2.imread(img_path)
    image = np.array(image, dtype=np.float32)

    annIds = coco.getAnnIds(imgIds=imgId)
    anns = coco.loadAnns(annIds)
    # anns是一个长度为N的列表，代表N个box，每个box是一个字典，
    # 包括'segmentation','area', 'iscrowd', 'image_id', 'bbox', 'category_id'
    segmentations = []
    bboxs = np.empty((len(anns), 4), dtype=np.int32)
    categories = []
    for idx, ann in enumerate(anns):
        bbox = np.array(ann['bbox'])
        # left top x, left top y, width, height -> xmin, ymin, xmax, ymax
        bbox_n = [bbox[0],
                  bbox[1],
                  bbox[0] + bbox[2],
                  bbox[1] + bbox[3]]

        bboxs[idx] = bbox_n
        categories.append(ann['category_id'])
        seg = ann['segmentation'][0]
        poly = np.array(seg).reshape((int(len(seg) / 2), 2))
        segmentations.append(poly)

    ########## data augmentation ########################

    raw_size = np.array((height, width), dtype=np.float32)

    image, bboxs, categories, segmentations = data_augmentation(image, raw_size, bboxs, categories, segmentations)

    ########## prepare classfication target and bbox regression target#############
    # plus the bg, there are 80 categories totally
    labels, targets, anchors = cls_target(image.shape, bboxs)

    ##########  prepare mask target ######################
    segmentation_mask = mask_target(image.shape, segmentations)

    ########## normalize the coordinate ######################
    for idx, target in enumerate(targets):
        targets[idx][[0, 2]] = target[[0, 2]] / input_shape[1]
        targets[idx][[1, 3]] = target[[1, 3]] / input_shape[0]


    for idx, one_kind_anchor in enumerate(anchors):
        anchors[idx][:, [0, 2]] = one_kind_anchor[:, [0, 2]] / input_shape[1]
        anchors[idx][:, [1, 3]] = one_kind_anchor[:, [1, 3]] / input_shape[0]

    ########## concat the rpn_binary_gt  rpn_bbox_gt  anchors ######################
    concat_labels = np.concatenate(labels)
    concat_targets = np.concatenate(targets)
    concat_anchors = np.concatenate(anchors)



    categories = np.array(categories, dtype=np.int32)

    # print(categories.dtype)

    #      image, gt_boxes, class_ids   input_gt_mask,     rpn_binary_gt  rpn_bbox_gt  anchors
    return image, bboxs,    categories, segmentation_mask, concat_labels,        concat_targets,     concat_anchors

















if __name__ == '__main__':
    data = producer()
    print(data)