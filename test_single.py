import copy
import time
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from common.visualizer import Visualizer
from common import html,util


from common.fileutils import makedirs
from common.imutils import imshowpts
from common.util import tensor2iuv
from test import sm_test_run

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 0   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.testset = opt.target_id
    opt.use_data_filters = False

    opt.DEBUG_MODE = True

    checkpoints_dir = opt.checkpoints_dir
    name = opt.name

    target_list = [2] if opt.train_list is None else opt.train_list
    source_list = [4]

    TARGET_LIST = [f'{f:04}' for f in target_list] #0,1,2
    SOURCE_LIST = [f'{f:04}' for f in source_list]
    for target_id in TARGET_LIST:
        for source_id in SOURCE_LIST :
            opt.source_id = source_id
            opt.target_id = target_id
            #opt.name = os.path.join(name , f'T{opt.target_id}')
            sm_test_run(opt,outvis='fake_B0')

