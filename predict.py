import copy
import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from common import html,util


from common.fileutils import makedirs
from options.configs import configs_loader
import numpy as np

def rescale_iuv(im):
    iuva = np.array(im)
    iuva[:, :, 2] = iuva[:, :, 2] / 10
    return np.uint8(iuva)

def source_to_target(dataset,outroot,source_obj):
    source_side = 'A' if source_obj == 1 else 'B'
    target_side = 'B' if source_obj == 1 else 'A'

    outvis = f'fake_{target_side}0'
    model = create_model(opt)

    for i, data in enumerate(dataset):
        model.set_input(data)
        model.test()
        visuals = model.get_current_visuals()
        img_path = model.get_image_paths(source_side)
        out_path = os.path.join(outroot,img_path[-1][0])
        makedirs(out_path)

        print(f'{i:04}: process image... {img_path[-1][0]}-->{out_path}')
        image_numpy = rescale_iuv(visuals[outvis])

        assert(out_path != img_path)
        util.save_image(image_numpy, out_path)


def predict_stage(opt_,data_cfgs) :
    opt = copy.deepcopy(opt_)
    data_loader = CreateDataLoader(opt,data_cfgs)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    #outroot = data_cfgs['OUTDIR']
    #outvis = data_cfgs['OUTVIS']

    obj1_name = data_cfgs['DATASET']['OBJ1']['NAME']
    obj2_name = data_cfgs['DATASET']['OBJ2']['NAME']
    outroot = os.path.join(opt.results_dir, opt.name,f'ep_{int(opt.which_epoch):05}')

    source_to_target(dataset, os.path.join(outroot,f'T{obj2_name}'), source_obj=1)
    source_to_target(dataset, os.path.join(outroot,f'T{obj1_name}'), source_obj=2)

if __name__ == '__main__':
    opt = TestOptions().parse()
    opt.nThreads = 0
    opt.batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.testset = opt.target_id
    opt.use_data_filters = False
    checkpoints_dir = opt.checkpoints_dir
    name = opt.name

    data_cfgs = configs_loader(opt.datafile)
    predict_stage(opt,data_cfgs)
