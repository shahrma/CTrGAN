import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import torch

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from common.imutils import imshowpts
from common.fileutils import get_dirs_list
import json
import yaml

from data.datautils import _makepath,resize_img,prepare_images

MISSING_VALUE = -1
class UnalignedPersonDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.dataparams = opt.dataroot
        self.target_id = opt.target_id
        self.source_id = opt.source_id
        self.remove_target_from_source = True
        self.load_keypoints = True

        with open(self.dataparams, 'r') as stream:
            pdata = yaml.safe_load(stream)
            pdata = pdata['DATASET']

        self.pdata = pdata
        self.dataroot = pdata['DATA_ROOT']

        tobj = [self.target_id]
        if opt.source_id != '.' :
            sobj = [opt.source_id]
        else :
            sobj = [i for i in pdata['OBJECTS'] if i not in tobj ]

        if hasattr(opt,'use_filters') and opt.use_filters:
            filters = pdata['FILTERS']
        else :
            filters = None

        self.A_paths = _makepath(self.dataroot,sobj, itype=pdata['RGB_DIR'],filters=filters)
        self.AP_paths = _makepath(self.dataroot,sobj, itype=pdata['IUV_DIR'],filters=filters)
        self.B_paths = _makepath(self.dataroot,tobj, itype=pdata['RGB_DIR'],filters=filters)
        self.BP_paths = _makepath(self.dataroot,tobj, itype=pdata['IUV_DIR'],filters=filters)


        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.BP_paths = sorted(self.BP_paths)
        self.AP_paths = sorted(self.AP_paths)

        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.AP_size = len(self.AP_paths)
        self.BP_size = len(self.BP_paths)

        if self.load_keypoints :
            self.AK_paths = _makepath(self.dataroot,sobj, itype=pdata['KPS_DIR'],images_only=False,filters=filters)
            self.BK_paths = _makepath(self.dataroot, tobj, itype=pdata['KPS_DIR'], images_only=False,filters=filters)
            self.BK_paths = sorted(self.BK_paths)
            self.AK_paths = sorted(self.AK_paths)

        # self.transform  = get_transform(opt)
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5, 0.5))] #SM added layer for alpha
        self.transform = transforms.Compose(transform_list)

        self.iuv_mode = opt.iuv_mode


    def get_dataroot(self):
        return self.dataroot


    def get_pose(self,filepath):
        with open(filepath) as f:
            kps = json.load(f)
        keypoints_x = kps['people'][0]['pose_keypoints_2d'][0::3]
        keypoints_y = kps['people'][0]['pose_keypoints_2d'][1::3]
        keypoints_s = kps['people'][0]['pose_keypoints_2d'][2::3]

        kp_array = np.concatenate((np.expand_dims(keypoints_y, -1),np.expand_dims(keypoints_x, -1)), axis=1)

        return kp_array

    def cords_to_map(self,cords, img_size = (256, 256), sigma=12):
        result = np.zeros(img_size + cords.shape[0:1], dtype='float32')
        for i, point in enumerate(cords):
            if point[0] == MISSING_VALUE or point[1] == MISSING_VALUE:
                continue
            xx, yy = np.meshgrid(np.arange(img_size[1]), np.arange(img_size[0]))
            result[..., i] = np.exp(-((yy - point[0]) ** 2 + (xx - point[1]) ** 2) / (2 * sigma ** 2))
        return result


    def __getitem__(self, index):
        index_A = index % self.A_size
        A_path = self.A_paths[index_A]
        AP_path = self.AP_paths[index_A]

        if self.opt.serial_batches:
            index_B = index % self.B_size
        else:
            index_B = random.randint(0, self.B_size - 1)

        B_path = self.B_paths[index_B]
        BP_path = self.BP_paths[index_B]

        # print('(A, B) = (%d, %d)' % (index_A, index_B))

        # read the triplet from A and B --
        #print(A_path,AP_path)
        #print(B_path,BP_path)
        A_img = Image.open(A_path).convert('RGBA')  #SM - changed to RGBA (instead of RGB)
        B_img = Image.open(B_path).convert('RGBA')  #SM - changed to RGBA (instead of RGB)
        AP_img = Image.open(AP_path).convert('RGBA')  # SM - changed to RGBA (instead of RGB)
        BP_img = Image.open(BP_path).convert('RGBA')  # SM - changed to RGBA (instead of RGB)

        # get the triplet from A
        A_img = resize_img(A_img, self.opt.loadSize, self.transform, itype='img')
        A0,A_offset = prepare_images(A_img, self.opt.fineSize)

        ## -- get the triplet from B
        B_img = resize_img(B_img, self.opt.loadSize, self.transform, itype='img')
        B0,B_offset = prepare_images(B_img, self.opt.fineSize)

        ## -- get the triplet from AP and BP
        AP_img = resize_img(AP_img, self.opt.loadSize, self.transform, itype=self.iuv_mode)
        AP0,_ = prepare_images(AP_img, self.opt.fineSize,offset=A_offset)

        BP_img = resize_img(BP_img, self.opt.loadSize, self.transform, itype=self.iuv_mode)
        BP0,_ = prepare_images(BP_img, self.opt.fineSize,offset=B_offset)

        if self.load_keypoints :
            AK_path = self.AK_paths[index_A]
            BK_path = self.BK_paths[index_B]

            AK_img = self.cords_to_map(self.get_pose(AK_path))
            BK_img = self.cords_to_map(self.get_pose(BK_path))

            AK_img = resize_img(AK_img, self.opt.loadSize, self.transform, itype='multi')
            AK0,_ = prepare_images(AK_img, self.opt.fineSize)

            BK_img = resize_img(BK_img, self.opt.loadSize, self.transform, itype='multi')
            BK0,_ = prepare_images(BK_img, self.opt.fineSize)
        else :
            AK0 = []
            BK0 = []

        #######
        input_nc = self.opt.input_nc
        output_nc = self.opt.output_nc

        return {'A0': A0,'B0': B0,  'AP0': AP0, 'BP0': BP0, 'AK0': AK0, 'BK0': BK0,
                'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedPersonDataset'


if __name__ == '__main__':
    pass

    '''
    def filter_dirs(rootpath ,dirslist,filters) :
        out = []
        for f in dirslist :
            sf = f.replace(rootpath, '').split('/')[1:]
            check = True
            for idx,_ in enumerate(filters) :
                if sf[idx] not in filters[idx] :
                    check = False

            if check :
                out.append(f)

        return sorted(out)

    def _makepath(dataroot, objs, itype = None,images_only=True,filters=None) :
        paths = []
        for item in objs :
            dirslist = get_dirs_list(os.path.join(dataroot,itype,item))
            if filters is not None :
                dirslist = filter_dirs(os.path.join(dataroot, itype, item), dirslist, filters=filters)
            for currdir in dirslist:
                paths.extend(make_dataset(currdir,images_only=images_only))
        return paths


    def prepare_images(A_img, fineSize,offset=None):
        w_total = A_img.size(2)
        w = int(w_total)
        h = A_img.size(1)
        if offset is None :
            w_offset = random.randint(0, max(0, w - fineSize - 1))
            h_offset = random.randint(0, max(0, h - fineSize - 1))
        else :
            w_offset,h_offset = offset
        A0 = A_img[:, h_offset:h_offset + fineSize,
             w_offset:w_offset + fineSize]

        return A0,[w_offset,h_offset]

    def split_hm(l,nl = 25) :
        out = np.zeros((l.shape[0],l.shape[1],nl))
        ul = np.unique(l)
        for il in ul[1:] :
            out[:,:,il] = (l==il).astype(int)
        return out

    def merge_hm(l) :
        out = np.zeros((l.shape[0],l.shape[1]))
        for il in range(l.shape[2]) :
            out[l[:,:,il].nonzero()] = il

        return out

    def merge_iuv(l, nl=25) :
        out = np.zeros((l.shape[0],l.shape[1]))
        for il in range(l.shape[2]) :
            out[l[:,:,il].nonzero()] = il

        return out

    def resize_img(img, loadSize, transform, itype='img'):
        if itype == 'img' :
            img = img.resize((loadSize, loadSize), Image.BICUBIC)
            img = transform(img)
        elif itype == 'multi':
            num_c = img.shape[-1]
            imgo = []
            for i in range(0,num_c) :
                img_ = Image.fromarray(img[:, :, i]).resize((loadSize, loadSize), Image.BICUBIC)
                imgo.append(torch.tensor(np.array(img_)).unsqueeze(0))
            img = torch.cat(imgo,dim=0)
        elif itype == 'iuv' :
            imgB = img.resize((loadSize, loadSize), Image.BICUBIC)
            imgN = img.resize((loadSize, loadSize), Image.NEAREST)
            img = np.array(imgB)
            img = transform(img)
            imgN = np.array(imgN)
            img[2,:,:] = torch.tensor(imgN[:, :, 2])*(2/255)-1

        elif itype == 'iuv_s' : #rescaled iuv instead of normed
            imgB = img.resize((loadSize, loadSize), Image.BICUBIC)
            imgN = img.resize((loadSize, loadSize), Image.NEAREST)
            img = torch.tensor(np.array(imgB))*(2/255)-1
            imgN = torch.tensor(np.array(imgN))*(2/255)-1
            img[:,:,2] = imgN[:,:,2]
            img = img.permute(2, 0, 1)
        elif itype == 'i_and_uv':
            imgB = img.resize((loadSize, loadSize), Image.BICUBIC)
            imgN = img.resize((loadSize, loadSize), Image.NEAREST)
            img = np.array(imgB)
            img = transform(img)
            imgN = np.array(imgN)
            img[2,:,:] = torch.tensor(imgN[:, :, 2])
            #img = PIL.Image.fromarray(np.uint8(img))
        elif itype == 'none':
                img = np.array(img)
                img = img.transpose(2, 0, 1)

        if itype in ['hm_iuv','iuv1']:
            imgB = img.resize((loadSize, loadSize), Image.BICUBIC)
            imgN = img.resize((loadSize, loadSize), Image.NEAREST)
            luv = np.array(imgB)[:, :, 0:2] * (2 / 255) - 1
            li = np.array(imgN)[:, :, 2]
            la = np.array(imgB)[:, :, 3] * (2 / 255) - 1

            if itype == 'iuv1' :
                li = ((10*li) * (2 / 255) - 1)
            elif itype == 'hm_iuv' :
                li = 2*split_hm(li, nl=25)-1

            if np.ndim(la) < 3 :
                la = np.expand_dims(la,axis=2)
            if np.ndim(li) < 3 :
                li = np.expand_dims(li,axis=2)

            img_ = torch.tensor(np.concatenate((luv,li,la),axis=2))
            img = img_.permute(2, 0, 1)

         return img
    '''

