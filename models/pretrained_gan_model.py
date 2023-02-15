import torch
import os
from . import networks

from models.transformer_pose import PoseTransformer
from data.datautils import get_centroids_path,load_centroids_images,get_dataparams

def freeze_model_weights(model):
    for param in model.parameters():
        param.requires_grad = False

def load_network(save_dir, network, network_label, epoch_label):
    save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
    save_path = os.path.join(save_dir, save_filename)
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path, map_location=torch.device('cpu'))
        network.load_state_dict(checkpoint)
        print(f'Loaded Network {save_path} !!!')
    else:
        print(f'Network {save_path} does not exist ...')


class PretrainedCyGANModel():
    def __init__(self, target_id,gpu_ids = [0],cycle_path=None):
        rootpath = cycle_path
        #rootpath = '../../pretrained_models/CyGAN'
        self.netG_A = networks.define_G(input_nc=4, output_nc=4, ngf=64, which_model_netG='unet_256', norm='instance',
                                   use_dropout=False, init_type='normal', gpu_ids=gpu_ids)

        save_dir = os.path.join(rootpath, f'T{target_id}', 'checkpoints')
        load_network(save_dir, self.netG_A, 'G_A', epoch_label=20)
        freeze_model_weights(self.netG_A)

    def apply(self,real_A0) :
        fake_B0 = self.netG_A(real_A0)
        return fake_B0




class PretrainedCTrGANModel():
    def __init__(self, yamlfile,target_id,gpu_ids = [0],use_transformer=True,cycle_path=None):
        rootpath = cycle_path
#        rootpath = '../../pretrained_models/CTrGAN'
        self.use_transformer = use_transformer

        self.netF_A = networks.define_G(input_nc=4, output_nc=4, ngf=16,which_model_netG='resnet_encoder',
                                        norm = 'instance', use_dropout = False, init_type = 'normal',
                                        gpu_ids=gpu_ids)
        self.netH_A = networks.define_G(input_nc=4, output_nc=4, ngf=16,which_model_netG= 'resnet_decoder',
                                        norm = 'instance', use_dropout = False, init_type = 'normal',
                                        gpu_ids=gpu_ids)

        self.netF_B = networks.define_G(input_nc=4, output_nc=4, ngf=16,which_model_netG='resnet_encoder',
                                        norm = 'instance', use_dropout = False, init_type = 'normal',
                                        gpu_ids=gpu_ids)
        self.Transformer_A = PoseTransformer(d_model=1024 * 4).cuda(gpu_ids[0])

        save_dir = os.path.join(rootpath, f'T{target_id}', 'checkpoints')

        load_network(save_dir, self.netF_A, 'F_A', epoch_label=20)
        load_network(save_dir, self.netH_A, 'H_A', epoch_label=20)
        load_network(save_dir, self.netF_B, 'F_B', epoch_label=20)
        load_network(save_dir, self.Transformer_A, 'Transformer_A', epoch_label=20)
        freeze_model_weights(self.netF_A)
        freeze_model_weights(self.netH_A)
        freeze_model_weights(self.netF_B)
        freeze_model_weights(self.Transformer_A)

        pdata = get_dataparams(yamlfile)
        self.B_keys = self.load_centroids(pdata['DATA_ROOT'],pdata['CENTROID_FILTERS'],pdata['CENTROIDS'],pdata['IUV_DIR'],item=target_id)
        self.B_keys = torch.Tensor(self.B_keys).cuda(gpu_ids[0]).type(torch.cuda.FloatTensor)

    def load_centroids(self,dataroot,filters,centroids_path,itype,item,iuv_mode = 'iuv1'):
        itemlist = get_centroids_path(dataroot, centroids_path, item, filters=filters, itype=itype)
        centroids = load_centroids_images(itemlist, iuv_mode=iuv_mode)
        return centroids

    def apply(self,real_A0) :
        if self.use_transformer:
            B_Keys = self.netF_B(self.B_keys)

        A_Query = self.netF_A(real_A0)
        A_Query = self.Transformer_A(B_Keys, A_Query) if self.use_transformer else A_Query
        fake_B0 = self.netH_A(A_Query)

        return fake_B0
