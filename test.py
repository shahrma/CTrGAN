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

def sm_test_run(opt_,outvis='fake_M0',source_id_model=None) :
    opt = copy.deepcopy(opt_)
    data_loader = CreateDataLoader(opt,data_cfgs)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataroot = os.path.join(dataset.dataset.get_dataroot(),'rgb',opt.source_id)
    name = opt.name
    if opt.single_source_mode :
        opt.name = os.path.join(name, f'T{opt.target_id}', source_id_model)
    else :
        opt.name = os.path.join(name, f'T{opt.target_id}')
    model = create_model(opt)

    results_dir = os.path.join(opt.results_dir, name, 'results',f'ep_{int(opt.which_epoch):05}','rgb', f'T{opt.target_id}',opt.source_id)

#    results_dir = os.path.join(opt.results_dir,opt.name , 'results',f'ep_{int(opt.which_epoch):05}', opt.source_id)
    # test
    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths()[0]  # ['S0']
        out_path = img_path.replace(dataroot,results_dir)
        makedirs(out_path)

        print(f'{i:04}: process image... {img_path}-->{out_path}')
        image_numpy = visuals[outvis]
        assert(out_path != img_path)
        util.save_image(image_numpy, out_path)

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 0   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.testset = opt.target_id
    opt.use_data_filters = False
    checkpoints_dir = opt.checkpoints_dir
    name = opt.name

    train_list = [0,1,2,3,4] if opt.train_list is None else opt.train_list
    source_list = range(0, 20) if opt.source_list is None else opt.source_list
    TARGET_LIST = [f'{f:04}' for f in train_list] #0,1,2
    SOURCE_LIST = [f'{f:04}' for f in source_list]
    for target_id in TARGET_LIST:
        for source_id in SOURCE_LIST :
            opt.source_id = source_id
            opt.target_id = target_id
            #opt.name = os.path.join(name , f'T{opt.target_id}')
            sm_test_run(opt,outvis='fake_B0',source_id_model= f'{opt.source_id}')

