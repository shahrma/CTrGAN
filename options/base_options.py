import argparse
import os
from common import util
import torch
import json
import shlex

class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', required=False, help='path to yaml file')

        self.parser.add_argument('--datarootA', type=str, default='.',  help='path to images - A path.')
        self.parser.add_argument('--datarootB', type=str, default='.', help='path to images - B path.')

        self.parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        self.parser.add_argument('--loadSize', type=int, default=286, help='scale images to this size')
        self.parser.add_argument('--fineSize', type=int, default=256, help='then crop to this size')
        self.parser.add_argument('--input_nc', type=int, default=4, help='# of input image channels')
        self.parser.add_argument('--output_nc', type=int, default=4, help='# of output image channels')
        self.parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in first conv layer')
        self.parser.add_argument('--npf', type=int, default=64, help='# of pred filters in first conv layer')
        self.parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in first conv layer')
        self.parser.add_argument('--which_model_netD', type=str, default='basic', help='selects model to use for netD')
        self.parser.add_argument('--which_model_netG', type=str, default='resnet_9blocks',
                                 help='selects model to use for netG')
        self.parser.add_argument('--which_model_netG_M', type=str, default='',
                                 help='netG_M for PPA architectures')
        self.parser.add_argument('--which_model_netP', type=str, default='unet_256',
                                 help='selects model to use for netG')
        self.parser.add_argument('--n_layers_D', type=int, default=3, help='only used if which_model_netD==n_layers')
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='experiment_name',
                                 help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--dataset_mode', type=str, default='unaligned',
                                 help='chooses how datasets are loaded. [unaligned | aligned | single]')
        self.parser.add_argument('--model', type=str, default='cycle_gan',
                                 help='chooses which model to use. cycle_gan, pix2pix, test')
        self.parser.add_argument('--which_direction', type=str, default='AtoB', help='AtoB or BtoA')
        self.parser.add_argument('--nThreads', default=0, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--norm', type=str, default='instance',
                                 help='instance normalization or batch normalization')
        self.parser.add_argument('--serial_batches', action='store_true',
                                 help='if true, takes images in order to make batches, otherwise takes them randomly')
        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8097, help='visdom port of the web display')
        self.parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"),
                                 help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')
        self.parser.add_argument('--resize_or_crop', type=str, default='resize_and_crop',
                                 help='scaling and cropping of images at load time [resize_and_crop|crop|scale_width|scale_width_and_crop]')
        self.parser.add_argument('--no_flip', action='store_true',
                                 help='if specified, do not flip the images for data augmentation')
        self.parser.add_argument('--init_type', type=str, default='normal',
                                 help='network initialization [normal|xavier|kaiming|orthogonal]')
        self.parser.add_argument('--use_pose_loss', action='store_true', help='use_pose_loss')
        self.parser.add_argument('--paired_offset', action='store_true', help='paired_offset - sync offset augmentation between A and B')

        self.parser.add_argument('--mask_mode', type=str, default='alpha', help='Gait Disicriminator mask mode (alpha vs. rgba)')
        self.parser.add_argument('--iuv_mode', type=str, default='iuv', help='iuv type - iuv [i-analog index] vs. iuv_s  [iuv - scale instead of norm] vs. iuv_digit [i-discret index,channel per index]')
        self.parser.add_argument('--use_sa', action='store_true', help='use self attention - only at reTransPP')
        self.parser.add_argument('--use_qsa', action='store_true', help='use self attention for q at decoder or only cross with encoder - only at reTransPP')
        self.parser.add_argument('--seq_len', default=3, type=int, help='# of quries in CTrGAN')
        self.parser.add_argument('--use_posenc', action='store_true', help='Whether using positional encoding at CTrGAN')

        self.parser.add_argument('--target_id', required=False, help='target id (from the yaml) the obj to generate to ')
        self.parser.add_argument('--source_id', type=str, default='.', help='CASIA-A SOURCE ID. (for unaligned_person dataset only .can be instead of yaml)')

        self.parser.add_argument('--use_fullseq', action='store_true',  help='Load subsequences or Full sequence (only at unaligned_sequence_dataset)')
        self.parser.add_argument('--use_filters', action='store_true',help='(only at unaligned_sequence_dataset)')
        self.parser.add_argument('--pretrained_cycle_model', type=str, default=None,
                                 help='pretrained_cycle_model. Options : None / CyGAN / CTrGAN')
        self.parser.add_argument('--direct_mode', type=str, default='P2A',
                                 help='Needed especially for CTrGAN. Two options, P2A (at train phase) or P2P2A (at inference phase)')
        self.parser.add_argument('--cycle_path', type=str, default=None,
                                 help='directory from where to load cycle (Internatls dirs are T000,T001, etc.)')
        self.parser.add_argument('--train_list', type=int,metavar='N',nargs='+', default=None, help='Which targets to train 0-4')
        self.parser.add_argument('--source_list', type=int, metavar='N', nargs='+', default=None, help='For One to One mode - Which sources to use at train')

        self.parser.add_argument('--DEBUG_MODE', action='store_true', help='CTrGAN - saves attention maps')

        self.initialized = True

    def parse(self,argstr=None):
        if not self.initialized:
            self.initialize()
        if argstr is None :
            self.opt = self.parser.parse_args()
        else :
            self.opt = self.parser.parse_args(shlex.split(argstr))

        self.opt.isTrain = self.isTrain  # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        expr_dir = os.path.join(self.opt.checkpoints_dir,'opts', self.opt.name)
        util.mkdirs(expr_dir)
        if self.isTrain :
            file_name = os.path.join(expr_dir, 'train_opt.txt')
            jfile_name = os.path.join(expr_dir, 'train_opt.json')
        else :
            util.mkdirs(os.path.join(expr_dir, f'{self.opt.testset}'))
            file_name = os.path.join(expr_dir, f'{self.opt.testset}','test_opt.txt')
            jfile_name = os.path.join(expr_dir, f'{self.opt.testset}','test_opt.json')

        with open(file_name, 'wt') as opt_file:
            opt_file.write('------------ Options -------------\n')
            for k, v in sorted(args.items()):
                opt_file.write('%s: %s\n' % (str(k), str(v)))
            opt_file.write('-------------- End ----------------\n')

        with open(jfile_name, 'w') as opt_file:
            json.dump(args, opt_file)

        return self.opt
