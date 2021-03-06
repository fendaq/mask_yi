import tensorflow as tf
import numpy as np
import os

# common
dataDir = '.'
train_type = 'val2017'
trainImage_path = '{}/cocodata/image'.format(dataDir)
train_annFile = '{}/cocodata/annotations/instances_{}.json'.format(dataDir,train_type)
checkpoint_path = 'checkpoints'
result_pic = 'result_pic' # 输出图片的地址
COCO_WEIGHTS_PATH = os.path.join(dataDir, 'pre_trained_weights', "mask_rcnn_coco.h5")  # 预训练模型所在地址
MASK_THRESH = 0.5 # mask的阈值，在测试时候对于输出的mask，大于该阈值的判为True，否则为False
restore = False # 是否接着训练
NUM_TEST_IMAGE = 100 # 测试图片的张数
max_steps = 100000
save_checkpoint_steps = 100 # 每100步保存一下ckeckpoints
save_summary_steps = 100 # 每100步保存一下summary
learning_rate = 0.001
DTYPE = tf.float32
moving_average_decay = 0.9
# train
input_shape = (512, 512) # （height，width）
random_resize = [0.5, 1, 1.5, 2]
pixel_average = [102.9801, 115.9465, 122.7717]
feat_strides = np.array([128, 64, 32, 16, 8])
positive_ratio = 0.25
posi_anchor_thresh = 0.5  # anchor 大于为正值
neg_anchor_thresh = 0.1  # anchor 小于为负值
batch_anchor_num = 128  # 每轮正值和负值总数
batch_size = 1
anchor_ratios = [1, .5, 2]
anchor_scales = [64, 128, 256]
anchor_per_location = len(anchor_ratios) * len(anchor_scales)

# 非极大值抑制以后，保留的roi的个数 (training and inference)
POST_NMS_ROIS_TRAINING = 2000
POST_NMS_ROIS_INFERENCE = 1000

RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2], dtype=np.float32)
RPN_NMS_THRESHOLD = 0.7  # 非极大值抑制的iou阈值
USE_MINI_MASK = True # 使用迷你mask，以节省内存
MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
# todo 保留上面，频道不需要
IMAGE_SHAPE = input_shape + (3,)  # 输入的图片的裁剪大小,通道是3

# 喂给网络的roi个数，原文里面是512，并保证正负样本比例1:3。但是有时候没有足够的正样本数来保证，故暂且取200
# 可以把proposal的nms的阈值调高，使得这个数可以适当调高
TRAIN_ROIS_PER_IMAGE = 200

# 正例样本在总样本中的占比
ROI_POSITIVE_RATIO = 0.3333
# 根据论文图4右边，mask的大小为 28*28
MASK_SHAPE = [28, 28]  # 必须是下面的两倍
MASK_POOL_SIZE = (14, 14) # 必须是上面的一半
POOL_SIZE = (7, 7)  # ROI Pooling层的大小，一般是7*7
NUM_CLASSES = 21  # 图片分为多少个类别。20类正例加一类背景。一般来说，我建议先分为object/non-object，
# 然后对于object再分为20类，而不是全部混在一起进行分类

DETECTION_MIN_CONFIDENCE = 0.7  # 属于某一类比的置信度阈值
DETECTION_MAX_INSTANCE = 100  # 每一张图片里面，最多检测出的instance个数
DETECTION_NMS_THRESHHOLD = 0.3  # 同类别的检测的非极大值抑制阈值
BACKBONE_STRIDES = []

if __name__ == '__main__':
    print(os.getcwd())