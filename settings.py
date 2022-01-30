import logging
import os.path as osp
import time

# EVAL = True: just test, EVAL = False: train and eval
EVAL = False

# dataset can be 'MIRFlickr' or 'NUSWIDE'

DATASET = 'MIRFlickr'

if DATASET == 'MIRFlickr':
    LABEL_DIR = '/media/qd/copy/zyc/Cleared-Set/LAll/mirflickr25k-lall.mat'
    TXT_DIR = '/media/qd/copy/zyc/Cleared-Set/YAll/mirflickr25k-yall.mat'
    IMG_DIR = '/media/qd/copy/zyc/Cleared-Set/IAll/mirflickr25k-iall.mat'
    NUM_EPOCH = 200
    LR_IMG = 0.005
    LR_TXT = 0.005
    EVAL_INTERVAL = 1

if DATASET == 'NUSWIDE':
    LABEL_DIR = '/home/dejie/data/DJSRH/NUS-WIDE/nus-wide-tc10-lall.mat'
    TXT_DIR = '/home/dejie/data/DJSRH/NUS-WIDE/nus-wide-tc10-yall.mat'
    IMG_DIR = '/home/dejie/data/DJSRH/NUS-WIDE/nus-wide-tc10-iall.mat'
    NUM_EPOCH = 200
    LR_IMG = 0.005
    LR_TXT = 0.005
    EVAL_INTERVAL = 1

K = 1.5
ETA = 0.1
ALPHA = 0.9

BATCH_SIZE = 4
CODE_LEN = 128
MOMENTUM = 0.7
WEIGHT_DECAY = 5e-4

GPU_ID = 0
NUM_WORKERS = 16
EPOCH_INTERVAL = 2

lrf = 0.01
lrr = 0.001
epochs = 30

MODEL_DIR = './checkpoint'
weights = '/media/qd/copy/zyc/DSAH-master/model/vit_large_patch16_224_in21k.pth'
freezelayers = 'True'
device = 'cuda:0'

logger = logging.getLogger('train')
logger.setLevel(logging.INFO)
now = time.strftime("%Y%m%d%H%M%S", time.localtime(time.time()))
log_name = now + '_log.txt'
log_dir = './log'
txt_log = logging.FileHandler(osp.join(log_dir, log_name))
txt_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
txt_log.setFormatter(formatter)
logger.addHandler(txt_log)

stream_log = logging.StreamHandler()
stream_log.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
stream_log.setFormatter(formatter)
logger.addHandler(stream_log)

logger.info('--------------------------Current Settings--------------------------')
logger.info('EVAL = %s' % EVAL)
logger.info('DATASET = %s' % DATASET)
logger.info('CODE_LEN = %d' % CODE_LEN)
logger.info('GPU_ID =  %d' % GPU_ID)
logger.info('ALPHA = %.4f' % ALPHA)
logger.info('K = %.4f' % K)
logger.info('ETA = %.4f' % ETA)


logger.info('NUM_EPOCH = %d' % NUM_EPOCH)
logger.info('BATCH_SIZE = %d' % BATCH_SIZE)
logger.info('NUM_WORKERS = %d' % NUM_WORKERS)
logger.info('EPOCH_INTERVAL = %d' % EPOCH_INTERVAL)
logger.info('EVAL_INTERVAL = %d' % EVAL_INTERVAL)


logger.info('LR_IMG = %.4f' % LR_IMG)
logger.info('LR_TXT = %.4f' % LR_TXT)

logger.info('MOMENTUM = %.4f' % MOMENTUM)
logger.info('WEIGHT_DECAY = %.4f' % WEIGHT_DECAY)

logger.info('--------------------------------------------------------------------')
