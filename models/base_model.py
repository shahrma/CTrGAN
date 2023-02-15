import os
import torch


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name,'checkpoints')
        os.makedirs(self.save_dir, exist_ok=True)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_visuals_wandb(self):
        return []

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    def measure(self):
        return dict()

    def measure_weights(self,network):
        lweights = list()
        for name, m in network.named_parameters():
            data_m= m.data.mean().item()
            data_s = m.data.std().item()
            grad_m = m.grad.mean().item()
            grad_s = m.grad.std().item()
            lweights.append({'name': name, 'data': {'mean': data_m, 'std': data_s }, 'grad': {'mean': grad_m, 'std': grad_s}})
        return lweights

    def freeze_model_weights(self,model):
        for param in model.parameters():
            param.requires_grad = False

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(gpu_ids[0])

    # helper loading function that can be used by subclasses
    def  network_exists(self, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        return os.path.exists(save_path)

    # helper loading function that can be used by subclasses
    def  load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        if os.path.exists(save_path) :
            checkpoint =  torch.load(save_path, map_location=torch.device('cpu'))
            network.load_state_dict(checkpoint)
            print(f'Loaded network {save_path} !!!!')
        else :
            print(f'Network {save_path} does not exist ...')

    def save_checkpoint(self, module, module_name, epoch_label):
        pass
        '''
        save_filename = f'{epoch_label}_{module_name}.pth'
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(module, save_path)
        '''
    def load_checkpoint(self, module, module_name, epoch_label):
        pass
        '''
        save_filename = f'{epoch_label}_{module_name}.pth'
        save_path = os.path.join(self.save_dir, save_filename)
        if os.path.exists(save_path) :
            checkpoint = torch.load(save_path, map_location=torch.device('cpu'))
            module.load_state_dict(checkpoint)
        else:
            print(f'Module {save_path} does not exist ...')
        '''

    # update learning rate (called once every epoch)
    def update_learning_rate(self):
        for scheduler in self.schedulers:
            scheduler.step()
        lr = self.optimizers[0].param_groups[0]['lr']
        print('learning rate = %.7f' % lr)
