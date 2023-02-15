import torch
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import common.util as util
from utils.image_pool import ImagePool
from .base_model import BaseModel
from . import networks

from models.transformer_pose import PoseTransformer

from models.vgg_perceptual_loss import VGGPerceptualLoss


class iCTrGANModel(BaseModel):
    def name(self):
        return 'iCTrGANModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        self.adversarial_loss_p = opt.adversarial_loss_p
        self.freeze_cycle = False if not self.isTrain else opt.freeze_cycle
        self.freeze_direct = False if not self.isTrain else opt.freeze_direct
        self.DEBUG_MODE = False if not hasattr(opt,'DEBUG_MODE') else opt.DEBUG_MODE

        nb =  opt.batchSize
        seq_len = 3 if not hasattr(opt,'seq_len') else opt.seq_len
        size = opt.fineSize
        self.input_A0 = self.Tensor(seq_len, opt.input_nc, size, size)
        self.input_B0 = self.Tensor(seq_len, opt.input_nc, size, size)

        self.input_M0 = self.Tensor(seq_len, opt.output_nc, size, size)
        self.input_S0 = self.Tensor(seq_len, opt.output_nc, size, size)


        # load/define networks
        # The naming conversion is different from those used in the paper
        # Code (paper): G_A (G), G_B (F), D_A (D_Y), D_B (D_X)

        self.netF_A = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, 'resnet_encoder', opt.norm, not opt.no_dropout, opt.init_type,
                                        self.gpu_ids)
        self.netH_A = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, 'resnet_decoder', opt.norm, not opt.no_dropout, opt.init_type,
                                        self.gpu_ids)
        self.netF_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, 'resnet_encoder', opt.norm, not opt.no_dropout, opt.init_type,
                                        self.gpu_ids)
        self.netH_B = networks.define_G(opt.output_nc, opt.input_nc,
                                        opt.ngf, 'resnet_decoder', opt.norm, not opt.no_dropout, opt.init_type,
                                        self.gpu_ids)

        self.use_transformer = opt.use_sa
        use_q_self_attention = False if not hasattr(opt,'use_qsa') else opt.use_qsa
        self.Transformer_A = PoseTransformer(d_model = 1024*4,use_q_self_attention=use_q_self_attention).cuda(opt.gpu_ids[0])
        self.Transformer_B = PoseTransformer(d_model = 1024*4,use_q_self_attention=use_q_self_attention).cuda(opt.gpu_ids[0])


        self.Transformer_A.set_debug_mode(self.DEBUG_MODE)

        self.use_perceptual_loss = opt.use_perceptual_loss if hasattr(opt,'use_perceptual_loss') else False
        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,
                                            opt.which_model_netD,
                                            opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

            if self.use_perceptual_loss :
                self.perceptual_loss = VGGPerceptualLoss().cuda(self.gpu_ids[0])

        which_epoch = opt.which_epoch

        if not self.isTrain or opt.continue_train:
            self.load_network(self.netF_A, 'F_A', which_epoch)
            self.load_network(self.netF_B, 'F_B', which_epoch)
            self.load_network(self.netH_A, 'H_A', which_epoch)
            self.load_network(self.netH_B, 'H_B', which_epoch)
            self.load_network(self.Transformer_A, 'Transformer_A', which_epoch)
            self.load_network(self.Transformer_B, 'Transformer_B', which_epoch)
            if self.isTrain:
                self.load_network(self.netD_A, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        self.freeze_cycle = False
        if self.freeze_cycle :
            self.freeze_model_weights(self.netF_A)
            self.freeze_model_weights(self.netF_B)
            self.freeze_model_weights(self.netH_A)
            self.freeze_model_weights(self.netH_B)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netF_A.parameters(), self.netH_A.parameters() ,
                                                                self.netF_B.parameters(), self.netH_B.parameters(),
                                                                self.Transformer_A.parameters(), self.Transformer_B.parameters()),
                                                                lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        self.last_debug_name = None

    def debug_attention_maps(self,input):
        encoder_map0, encoder_map1 = self.Transformer_A.get_debug()
        if self.last_debug_name is not None:
            import os
            import pickle as pkl

            attention_dir = './output/attentions3/'
            os.makedirs(attention_dir,exist_ok=True)
            oh, ow = encoder_map0.shape

            with open(os.path.join(attention_dir,f'0_{self.last_debug_name}'), 'wb') as f:
                pkl.dump(encoder_map0, f)

            with open(os.path.join(attention_dir,f'1_{self.last_debug_name}'), 'wb') as f:
                pkl.dump(encoder_map1, f)

        self.last_debug_name = input['A_paths'][0][0].replace('../../datasets/CASIA/DatasetA/Crop256/rgb',
                                                              'debug').replace('/', '_').replace('.png', '.pkl')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A0 = input['AP0'][0]
        input_B0 = input['BP0'][0]

        input_S0 = input['A0'][0]
        input_M0 = input['B0'][0]
        self.imgidx = input['imgidx']
        new_seq = True if input['imgidx'] == 1 else False

        self.input_A0.resize_(input_A0.size()).copy_(input_A0)
        self.input_B0.resize_(input_B0.size()).copy_(input_B0)
        self.input_M0.resize_(input_M0.size()).copy_(input_M0)
        self.input_S0.resize_(input_S0.size()).copy_(input_S0)

        if new_seq :
            self.start_new_sequence(input)
            if self.DEBUG_MODE :
                self.debug_attention_maps(input)

        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def start_new_sequence(self,input):
        self.A_keys = input['A_keys'][0].to(self.input_A0.device).type(torch.cuda.FloatTensor)
        self.B_keys = input['B_keys'][0].to(self.input_B0.device).type(torch.cuda.FloatTensor)
        self.A_obj = input['A_obj']
        self.B_obj = input['B_obj']
        self.Transformer_A.start_new_sequence()
        self.Transformer_B.start_new_sequence()

    def forward(self):
        self.real_A0 = Variable(self.input_A0)
        self.real_B0 = Variable(self.input_B0)
        self.real_M0 = Variable(self.input_M0)
        self.real_S0 = Variable(self.input_S0)

    def test(self):
        real_A0 = Variable(self.input_A0)
        real_B0 = Variable(self.input_B0)

        if self.use_transformer :
            B_Keys = self.netF_B(self.B_keys)
            A_Keys = self.netF_A(self.A_keys)

        A_Query = self.netF_A(real_A0)
        A_Query = self.Transformer_A(B_Keys,A_Query) if self.use_transformer else A_Query
        fake_B0 = self.netH_A(A_Query)

        B_Query = self.netF_B(real_B0)
        B_Query = self.Transformer_B(A_Keys, B_Query) if self.use_transformer else B_Query
        fake_A0 = self.netH_B(B_Query)

        if not self.DEBUG_MODE :

            A_Query = self.netF_A(fake_A0)
            A_Query = self.Transformer_A(B_Keys,A_Query) if self.use_transformer else A_Query
            rec_B0 = self.netH_A(A_Query)

            B_Query = self.netF_B(fake_B0)
            B_Query = self.Transformer_B(A_Keys, B_Query) if self.use_transformer else B_Query
            rec_A0 = self.netH_B(B_Query)
        else :
            rec_A0 = fake_A0
            rec_B0 = fake_B0
        self.rec_B0 = rec_B0.data
        self.rec_A0 = rec_A0.data

        self.fake_B0 = fake_B0.data
        self.fake_A0 = fake_A0.data
        self.rec_B0 = fake_B0.data
        self.rec_A0 = fake_A0.data



    # get image paths
    def get_image_paths(self):
        return self.image_paths[-1]

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B0 = self.fake_B_pool.query(self.fake_B0)
        loss_D_A0 = self.backward_D_basic(self.netD_A, self.real_B0, fake_B0)

        self.loss_D_A = loss_D_A0.item()

    def backward_D_B(self):
        fake_A0 = self.fake_A_pool.query(self.fake_A0)
        loss_D_B0 = self.backward_D_basic(self.netD_B, self.real_A0, fake_A0)

        self.loss_D_B = loss_D_B0.item()

    def backward_G(self):
        lambda_idt = self.opt.identity
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B
        if self.use_transformer :
            B_Keys = self.netF_B(self.B_keys)
            A_Keys = self.netF_A(self.A_keys)

        # Identity loss
        if lambda_idt > 0:
            # G_A should be identity if real_B is fed.
            A_Query = self.netF_A(self.real_B0)
            A_Query = self.Transformer_A(B_Keys, A_Query) if self.use_transformer else A_Query
            idt_A0 = self.netH_A(A_Query)
            loss_idt_A = (self.criterionIdt(idt_A0, self.real_B0)) * lambda_B * lambda_idt
            # G_B should be identity if real_A is fed.
            B_Query = self.netF_B(self.real_A0)
            B_Query = self.Transformer_B(A_Keys, B_Query) if self.use_transformer else B_Query
            idt_B0 = self.netH_B(B_Query)
            loss_idt_B = (self.criterionIdt(idt_B0, self.real_A0)) * lambda_A * lambda_idt

            self.idt_A = idt_A0.data
            self.idt_B = idt_B0.data
            self.loss_idt_A = loss_idt_A.item()
            self.loss_idt_B = loss_idt_B.item()
        else:
            loss_idt_A = 0
            loss_idt_B = 0
            self.loss_idt_A = 0
            self.loss_idt_B = 0

        # GAN loss D_A(G_A(A))
        A_Query = self.netF_A(self.real_A0)
        A_Query = self.Transformer_A(B_Keys,A_Query) if self.use_transformer else A_Query
        fake_B0 = self.netH_A(A_Query)
        pred_fake_B = self.netD_A(fake_B0)
        loss_G_A0 = self.criterionGAN(pred_fake_B, True)

        # GAN loss D_B(G_B(B))
        B_Query = self.netF_B(self.real_B0)
        B_Query = self.Transformer_B(A_Keys, B_Query) if self.use_transformer else B_Query
        fake_A0 = self.netH_B(B_Query)
        pred_fake_A = self.netD_B(fake_A0)
        loss_G_B0 = self.criterionGAN(pred_fake_A, True)

            # Fwd cycle loss
        A_Query = self.netF_A(fake_A0)
        A_Query = self.Transformer_A(B_Keys,A_Query) if self.use_transformer else A_Query
        rec_B0 = self.netH_A(A_Query)
        loss_cycle_B0 = self.criterionCycle(rec_B0, self.real_B0) * lambda_B

        B_Query = self.netF_B(fake_B0)
        B_Query = self.Transformer_B(A_Keys, B_Query) if self.use_transformer else B_Query
        rec_A0 = self.netH_B(B_Query)
        loss_cycle_A0 = self.criterionCycle(rec_A0, self.real_A0) * lambda_A

        loss_G = (loss_G_A0 + loss_G_B0 +
                  loss_idt_A + loss_idt_B +
                   loss_cycle_A0 + loss_cycle_B0)

        if self.use_perceptual_loss:
            loss_G_perc_A0I = self.perceptual_loss(rec_A0[:,0:3] , self.real_A0[:,0:3])
            loss_G_perc_A0A = self.perceptual_loss(rec_A0[:,3:], self.real_A0[:, 3:])
            loss_G_perc_B0I = self.perceptual_loss(rec_B0[:,0:3] , self.real_B0[:,0:3])
            loss_G_perc_B0A = self.perceptual_loss(rec_B0[:,3:], self.real_B0[:, 3:])
            loss_G_perc = loss_G_perc_B0I + loss_G_perc_B0A + loss_G_perc_A0I + loss_G_perc_A0A

            loss_G += loss_G_perc

        loss_G.backward()

        self.fake_B0 = fake_B0.data
        self.fake_A0 = fake_A0.data
        self.rec_B0 = rec_B0.data
        self.rec_A0 = rec_A0.data

        self.pred_fake_B = pred_fake_B.data
        self.pred_fake_A = pred_fake_A.data

        self.loss_G_A = loss_G_A0.item()
        self.loss_G_B = loss_G_B0.item()

        self.loss_cycle_A = loss_cycle_A0.item()
        self.loss_cycle_B = loss_cycle_B0.item()

    def optimize_parameters(self):
        # forward
        self.forward()
        # G_A and G_B
        self.optimizer_G.zero_grad()
        self.backward_G()
        self.optimizer_G.step()


        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()
        # D_B
        self.optimizer_D_B.zero_grad()
        self.backward_D_B()
        self.optimizer_D_B.step()

    def get_current_errors(self):

        ret_errors = OrderedDict([('D_A', self.loss_D_A), ('G_A', self.loss_G_A),
                                  ('G_B', self.loss_G_B), ('D_B', self.loss_D_B),
                                  ('Cyc_A', self.loss_cycle_A), ('Cyc_B', self.loss_cycle_B) ])
        for i,scheduler in enumerate(self.schedulers) :
            ret_errors[f'lr_{i}'] = scheduler.get_last_lr()[0]

        if self.opt.identity > 0.0:
            ret_errors['idt_A'] = self.loss_idt_A
            ret_errors['idt_B'] = self.loss_idt_B

        return ret_errors

    def get_current_visuals(self):
        real_A0 = util.tensor2iuv(self.input_A0)
        fake_B0F = util.tensor2iuv(self.fake_B0[0:1])
        fake_B0L = util.tensor2iuv(self.fake_B0[-1:])
        real_B0 = util.tensor2iuv(self.input_B0)
        fake_A0F = util.tensor2iuv(self.fake_A0[0:1])
        fake_A0L = util.tensor2iuv(self.fake_A0[-1:])
        rec_B0 = util.tensor2iuv(self.rec_B0)
        rec_A0 = util.tensor2iuv(self.rec_A0)

        fake_B0_I = util.tensor2im(self.fake_B0[0:1,2,:,:].unsqueeze(0))
        fake_A0_I = util.tensor2im(self.fake_A0[0:1,2,:,:].unsqueeze(0))

        real_M0 = util.tensor2im(self.input_M0)
        real_S0 = util.tensor2im(self.input_S0)

        ret_visuals = OrderedDict([('real_A0', real_A0), ('fake_B0F', fake_B0F), ('fake_B0', fake_B0L),('real_S0', real_S0),
                                   ('real_B0', real_B0), ('fake_A0F', fake_A0F), ('fake_A0', fake_A0L),
                                   ('fake_B0_I',fake_B0_I), ('fake_A0_I',fake_A0_I),
                                   ('real_M0', real_M0) , ('rec_B0',rec_B0),('rec_A0',rec_A0)])

        if self.opt.isTrain :
            ret_visuals['pred_fake_B'] = util.tensor2im(self.pred_fake_B)
            ret_visuals['pred_fake_A'] = util.tensor2im(self.pred_fake_A)

        if self.opt.isTrain and self.opt.identity > 0.0:
            ret_visuals['idt_A'] = util.tensor2iuv(self.idt_A)
            ret_visuals['idt_B'] = util.tensor2iuv(self.idt_B)
        return ret_visuals

    def get_current_visuals_wandb(self):
        real_A0 = util.tensor2iuv(self.input_A0)
        fake_B0 = util.tensor2iuv(self.fake_B0)
        real_B0 = util.tensor2iuv(self.input_B0)
        fake_A0 = util.tensor2iuv(self.fake_A0)

        fake_B0_I = util.tensor2im(self.fake_B0[:,2,:,:].unsqueeze(0))
        fake_A0_I = util.tensor2im(self.fake_A0[:,2,:,:].unsqueeze(0))
        rec_B0 = util.tensor2iuv(self.rec_B0)
        rec_A0 = util.tensor2iuv(self.rec_A0)

        real_M0 = util.tensor2im(self.input_M0)
        real_S0 = util.tensor2im(self.input_S0)

        ret_visuals = {'real' : OrderedDict([('real_A0', real_A0),('real_S0', real_S0), ('real_B0', real_B0), ('real_M0', real_M0)]),
                       'fake' : OrderedDict([('fake_A0', fake_A0), ('fake_B0', fake_B0),('fake_B0_I',fake_B0_I), ('fake_A0_I',fake_A0_I)]),
                       'others' :OrderedDict([('rec_B0',rec_B0),('rec_A0',rec_A0)])}

        if self.opt.isTrain :
            ret_visuals['others']['pred_fake_B'] = util.tensor2im(self.pred_fake_B)
            ret_visuals['others']['pred_fake_A'] = util.tensor2im(self.pred_fake_A)

        if self.opt.isTrain and self.opt.identity > 0.0:
            ret_visuals['others']['idt_A'] = util.tensor2iuv(self.idt_A)
            ret_visuals['others']['idt_B'] = util.tensor2iuv(self.idt_B)

        return ret_visuals


    def save(self, label):
        self.save_network(self.netF_A, 'F_A', label, self.gpu_ids)
        self.save_network(self.netH_A, 'H_A', label, self.gpu_ids)
        self.save_network(self.netF_B, 'F_B', label, self.gpu_ids)
        self.save_network(self.netH_B, 'H_B', label, self.gpu_ids)
        self.save_network(self.Transformer_A, 'Transformer_A', label, self.gpu_ids)
        self.save_network(self.Transformer_B, 'Transformer_B', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)

        for i, scheduler in enumerate(self.schedulers):
            self.save_checkpoint(scheduler, f'scheduler{i}', label)

    def measure(self):
        m = dict()
        m['F_A'] = self.measure_weights(self.netF_A)
        m['H_A'] = self.measure_weights(self.netH_A)
        m['F_B'] = self.measure_weights(self.netF_B)
        m['H_B'] = self.measure_weights(self.netH_B)
        m['T_A'] = self.measure_weights(self.Transformer_A)
        m['T_B'] = self.measure_weights(self.Transformer_B)
        m['D_A'] = self.measure_weights(self.netD_A)
        m['D_B'] = self.measure_weights(self.netD_B)
        o = dict()
        for net in m :
            dm, ds, gm, gs = list(), list(), list(), list()
            for i in m[net]:
                dm.append(i['data']['mean'])
                ds.append(i['data']['std'])
                gm.append(i['grad']['mean'])
                gs.append(i['grad']['std'])
            o[net] = {'dm': dm, 'ds': ds, 'gm': gm, 'gs': gs}

        return m,o
