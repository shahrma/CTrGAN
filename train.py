import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from common.visualizer import Visualizer
from common.fileutils import VariablesLogger
import wandb

def log_wandb_images(wandb, cvis):
    if len(cvis) == 0:
        return
    for item in cvis:
        lcvis_ = [wandb.Image(cvis[item][i], caption=i) for i in cvis[item]]
        wandb.log({f"samples_{item}": lcvis_})


def train_stage(opt):
    wandb_mode = opt.wandb_mode  # "disabled" if True else "online"

    run_id = opt.name.replace('/', '_')
    run = wandb.init(project='fake_vp', entity='smahpod', id=run_id, resume='allow', mode=wandb_mode)
    run.name = opt.name
    run.save()
    wbcfg = wandb.config
    wbcfg.update(opt, allow_val_change=True)
    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    dataset_size = len(data_loader)
    print('#training images = %d' % dataset_size)

    model = create_model(opt)
    visualizer = Visualizer(opt)
    total_steps = 0
    vlog = VariablesLogger(os.path.join(opt.checkpoints_dir,opt.name,'vars'))
    last_epoch = vlog.load_param('last_epoch',0)

    # train stage
    for epoch in range(last_epoch, opt.niter + opt.niter_decay + 1):
        epoch_start_time = time.time()
        epoch_iter = 0

        for i, data in enumerate(dataset):
            iter_start_time = time.time()
            visualizer.reset()
            total_steps += opt.batchSize
            epoch_iter += opt.batchSize
            model.set_input(data)
            model.optimize_parameters()

            if total_steps % opt.display_freq == 0:
                save_result = total_steps % opt.update_html_freq == 0
                visualizer.display_current_results(model.get_current_visuals(), epoch, save_result)

            if total_steps % opt.print_freq == 0:
                errors = model.get_current_errors()
                t = (time.time() - iter_start_time) / opt.batchSize
                visualizer.print_current_errors(epoch, epoch_iter, errors, t)

                wandb.log(errors)
                if opt.display_id > 0:
                    visualizer.plot_current_errors(epoch, float(epoch_iter) / dataset_size, opt, errors)

            if total_steps % opt.save_latest_freq == 0:
                print('saving the latest model (epoch %d, total_steps %d)' %
                      (epoch, total_steps))
                model.save('latest')

        #errors['epoch'] = epoch


        cvis = model.get_current_visuals_wandb()
        # log_wandb_images(wandb, cvis)

        if epoch % opt.save_epoch_freq == 0:
            print('saving the model at the end of epoch %d, iters %d' %
                  (epoch, total_steps))
            model.save('latest')
            model.save(epoch)

        vlog.save_param('last_epoch', epoch)
        print('End of epoch %d / %d \t Time Taken: %d sec' %
              (epoch, opt.niter + opt.niter_decay, time.time() - epoch_start_time))
        model.update_learning_rate()


import os


def run_series(opt, target_list, source_list):
    epoch = opt.niter + opt.niter_decay
    if opt.train_mode == 'train' or opt.train_mode == 'full':
        TARGET_LIST = [f'{i:04}' for i in target_list]
        name = opt.name
        for target_id in TARGET_LIST:
            opt.target_id = target_id
            if source_list is None :
                opt.name = os.path.join(name , f'T{opt.target_id}')
                train_stage(opt)
            else :
                SOURCE_LIST = [f'{i:04}' for i in source_list]
                for source_id in SOURCE_LIST:
                    opt.source_id = source_id
                    opt.name = os.path.join(name, f'T{opt.target_id}', f'{opt.source_id}' )
                    train_stage(opt)



if __name__ == '__main__':
    opt = TrainOptions().parse()
    # run_series(opt,ids = [0, 2, 3])
    train_list = [0, 1, 2, 3, 4] if opt.train_list is None else opt.train_list
    run_series(opt, target_list=train_list, source_list = opt.source_list)
