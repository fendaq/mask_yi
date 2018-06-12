import time
import shutil
import cv2
import numpy as np
import tensorflow as tf
import os
import urllib.request
import config
from model import build_model
from input.producer import producer

# os.environ["CUDA_VISIBLE_DEVICES"] = "0"
weights_path = config.COCO_WEIGHTS_PATH

def main(_):
    ##############下面，除了图片，所有坐标都是归一化坐标##################
    # 输入图片 [批数，高，宽，通道数]
    # TODO 输入数据的类型需要统一 noted by csh
    input_images = tf.placeholder(dtype=tf.float32, shape=(1, None, None, 3), name='input_images')
    # ground truth， [批数，MAX_GT_INSTANCES，4]
    gt_boxes = tf.placeholder(dtype=tf.float32, shape=(1, None, 4), name='gt_boxes')
    # 类别编号 [批数，MAX_GT_INSTANCES]
    class_ids = tf.placeholder(dtype=tf.int32, shape=(1, None), name='class_ids')
    # TODO 是否是 (1, None, None) ? noted by csh mask，[批数，MAX_GT_INSTANCES，高，宽]
    input_gt_mask = tf.placeholder(dtype=tf.float32, shape=(1, None, None, None), name='input_gt_mask')
    # 真实的anchor，[批数，anchor个数，标签]，其中1表示正例，0表示负例，-1表示不予考虑
    rpn_binary_gt = tf.placeholder(dtype=tf.int32, shape=(1, None, 1), name='rpn_binary_gt')
    # TODO 这里的log(h), log(w)是否是归一化后的? 真实anchor的回归真值 [批数，anchor个数，(dx, dy, log(h),log(w))]
    # TODO 暂时计算出的是 [批数，anchor个数，(xmin, ymin, xmax, ymax)] noted by csh
    rpn_bbox_gt = tf.placeholder(dtype=tf.float32, shape=(1, None, 4), name='rpn_bbox_gt')
    # anchor坐标，归一化坐标 [批数，个数，(x1, y1, x2, y2)]
    anchors = tf.placeholder(dtype=tf.float32, shape=(1, None, 4), name='anchors')

    if not tf.gfile.Exists(config.checkpoint_path):
        tf.gfile.MakeDirs(config.checkpoint_path)
    else:
        if not config.restore:
            tf.gfile.DeleteRecursively(config.checkpoint_path)
            tf.gfile.MakeDirs(config.checkpoint_path)

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(config.learning_rate, global_step, decay_steps=10000, decay_rate=0.94,
                                               staircase=True)
    tf.summary.scalar('lr', learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate)
    # 返回一个列表，包含[rpn_loss, proposal_loss, mask_loss, model_loss]
    rpn_loss, proposal_loss, mask_loss, model_loss = build_model(
        'training', input_images, gt_boxes, class_ids, input_gt_mask, rpn_binary_gt, rpn_bbox_gt, anchors)

    total_loss = model_loss + tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

    batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    gradient_op = opt.minimize(loss=total_loss, global_step=global_step)
    summary_op = tf.summary.merge_all()
    # 定义滑动平均对象
    variable_averages = tf.train.ExponentialMovingAverage(config.moving_average_decay, global_step)
    # 将该滑动平均对象作用于所有的可训练变量。tf.trainable_variables()以列表的形式返回所有可训练变量
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # 下面这两句话等价于 train_op = tf.group(variables_averages_op, apply_gradient_op, batch_norm_updates_op)
    with tf.control_dependencies([variables_averages_op, gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(config.checkpoint_path, tf.get_default_graph())

    init = tf.global_variables_initializer()




    next_batch = producer().make_one_shot_iterator().get_next()


    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        if config.COCO_WEIGHTS_PATH is not None and not config.restore:
            try:
                print("trying loading pre-trained model...")
                load_trained_weights(weights_path, sess, ignore_missing=True)
            except:
                raise 'loading pre-trained model failed,please check your pretrained ' \
                      'model {:s}'.format(config.COCO_WEIGHTS_PATH)

        # 如果是从原来的模型中接着训练，就不需要sess.run(tf.global_variables_initializer())
        if config.restore:
            print('continue training from previous checkpoint')
            ckpt = tf.train.latest_checkpoint(config.checkpoint_path)
            saver.restore(sess, ckpt)
        else:
            sess.run(init)

        start = time.time()
        for step in range(config.max_steps):
            # TODO 数据获得
            # image, gt_boxes, class_ids   input_gt_mask,     rpn_binary_gt  rpn_bbox_gt  anchors
            image, bboxs, categories, segmentation_mask, concat_labels, concat_targets, concat_anchors = sess.run(next_batch)
            #
            # print(image.shape)
            # print(bboxs.shape)
            # print(categories.shape)
            # print(segmentation_mask.shape)
            # print(concat_labels.shape)
            # print(concat_targets.shape)
            # print(concat_anchors)
            #
            #
            # data = [None, None, None, None, None, None, None]
            # image = cv2.imread('E:\\DLnet\\ctpn\\dataset\\for_test\\TB1_hpqJFXXXXX2aFXXunYpLFXX.jpg')
            # image = cv2.resize(image, (1024, 1024))
            # # image = cv2.imread('tu2.jpg')
            # data[0] = image[np.newaxis, ...]
            #
            # data[1] = [[[0.0, 0.2, 0.5, 0.7]]]
            # data[2] = [[1]]
            # data[3] = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)[np.newaxis, ...][np.newaxis, ...]
            # # print(data[3].shape)
            # data[4] = [[[0]]]
            # data[5] = [[[0, 0, 1, 1]]]
            # data[6] = [[[0.0, 0.0, 0.5, 0.5]]]
            ml, tl, _ = sess.run([model_loss, total_loss, train_op], feed_dict={input_images: image,
                                                                                gt_boxes: bboxs,
                                                                                class_ids: categories,
                                                                                input_gt_mask: segmentation_mask,
                                                                                rpn_binary_gt: concat_labels,
                                                                                rpn_bbox_gt: concat_targets,
                                                                                anchors: concat_anchors})
            if np.isnan(tl):
                print('Loss diverged, stop training')
                break

            if step % 10 == 0:
                avg_time_per_step = (time.time() - start) / 10
                start = time.time()
                print('Step {:06d}, model loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step'.format(step, ml, tl,
                                                                                                      avg_time_per_step))

            if step % config.save_checkpoint_steps == 0:
                saver.save(sess, config.checkpoint_path + 'model.ckpt', global_step=global_step)

            if step % config.save_summary_steps == 0:
                _, tl, summary_str = sess.run([train_op, total_loss, summary_op], feed_dict={input_images: data[0],
                                                                                             gt_boxes: data[2],
                                                                                             class_ids: data[3],
                                                                                             input_gt_mask: data[4],
                                                                                             rpn_binary_gt: data[5],
                                                                                             rpn_bbox_gt: data[6],
                                                                                             anchors: data[7]})
                summary_writer.add_summary(summary_str, global_step=step)


def load_trained_weights(file_path, sess, ignore_missing=False):
    import h5py
    if h5py is None:
        raise ImportError('load_weights require h5py')
    data_dict = h5py.File(file_path, mode='r')
    for key in data_dict.keys():
        # with tf.variable_scope(key, reuse=True):
        for subkey in data_dict[key]:
            try:
                var = tf.get_variable(subkey)
                sess.run(var.assign(data_dict[key][subkey]))
                print("assign pretrain model "+subkey+ " to "+key)
            except ValueError:
                print("ignore "+ key)
                if not ignore_missing: # 有缺失项，但是又不去忽略，就报错
                    raise


def download_trained_weights(coco_model_path, verbose=1):
    if verbose > 0:
        print("Downloading pretrained model to " + coco_model_path + " ...")
    with urllib.request.urlopen(COCO_MODEL_URL) as resp, open(coco_model_path, 'wb') as out:
        shutil.copyfileobj(resp, out)
    if verbose > 0:
        print("... done downloading pretrained model!")

if __name__ == '__main__':

    COCO_MODEL_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"
    if not os.path.exists(weights_path):
        download_trained_weights(weights_path)
    tf.app.run(main)
