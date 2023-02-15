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
from random import randint

from data.datautils import resize_img,prepare_images,makeseq
from data.datautils import get_centroids_path,load_centroids_images

MISSING_VALUE = -1
class UnalignedSequenceDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.dataparams = opt.dataroot
        self.target_id = opt.target_id
        self.source_id = opt.source_id
        self.use_target_in_source = False if not hasattr(opt, 'use_target_in_source') else opt.use_target_in_source
        self.load_keypoints = True

        self.use_filters = True if not hasattr(opt, 'use_filters') else opt.use_filters
        self.use_fullseq = True if not hasattr(opt, 'use_fullseq') else opt.use_fullseq

        with open(self.dataparams, 'r') as stream:
            pdata = yaml.safe_load(stream)
            pdata = pdata['DATASET']

        self.pdata = pdata
        self.dataroot = pdata['DATA_ROOT']

        tobj = [self.target_id]
        if opt.source_id != '.' :
            sobj = [opt.source_id]
        else :
            if not self.use_target_in_source :
                sobj = [i for i in pdata['OBJECTS'] if i not in tobj ]
            else :
                sobj = [i for i in pdata['OBJECTS']]

        self.filters = pdata['FILTERS'] if self.use_filters  else None
        self.centroid_filters = pdata['CENTROID_FILTERS']

        self.sources = sorted(sobj)
        self.targets = sorted(tobj)

        self.A_paths,self.A_objid = makeseq(self.dataroot,sobj, itype=pdata['RGB_DIR'],filters=self.filters)
        self.AP_paths,_ = makeseq(self.dataroot,sobj, itype=pdata['IUV_DIR'],filters=self.filters)
        self.B_paths,self.B_objid = makeseq(self.dataroot,tobj, itype=pdata['RGB_DIR'],filters=self.filters)
        self.BP_paths,_ = makeseq(self.dataroot,tobj, itype=pdata['IUV_DIR'],filters=self.filters)

        self.A_size = sum([len(i) for i in self.A_paths])
        self.B_size = sum([len(i) for i in self.B_paths])

        if self.load_keypoints :
            self.AK_paths,_ = makeseq(self.dataroot,sobj, itype=pdata['KPS_DIR'],images_only=False,filters=self.filters)
            self.BK_paths,_ = makeseq(self.dataroot, tobj, itype=pdata['KPS_DIR'], images_only=False,filters=self.filters)

        # self.transform  = get_transform(opt)
        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5, 0.5))] #SM added layer for alpha
        self.transform = transforms.Compose(transform_list)

        self.iuv_mode = opt.iuv_mode

        self.centroids =  self.load_centroids(pdata['CENTROIDS'],pdata['IUV_DIR'])

        self.subseq_idx = 0
        self.packet_len = 3 if not hasattr(opt,'seq_len') else opt.seq_len

        if not self.use_fullseq :
            self.subseq_len = 30
            self.start_new_subseq()
        else :
            self.currA_seq = 0
            self.currB_seq = 0
            self.num_seq = len(self.A_paths)
            self.start_fullseq()

    def load_centroids(self,centroids_path,itype):
        centroids = {}

        for item in centroids_path:
            itemlist = get_centroids_path(self.dataroot, centroids_path, item, filters=self.centroid_filters, itype=itype)
            centroids[item] = load_centroids_images(itemlist, iuv_mode=self.iuv_mode)

        return centroids

    def get_max_centroids(self):
        return max([self.centroids[i].shape[0] for i in self.centroids])

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
    '''
    def get_data(self,I_path,P_path,K_path):
        I_img = Image.open(I_path).convert('RGBA')
        P_img = Image.open(P_path).convert('RGBA')

        # get the triplet from A
        I_img = resize_img(I_img, self.opt.loadSize, self.transform, itype='img')
        AI0 = prepare_images(I_img, self.opt.fineSize)

        ## -- get the triplet from AP and BP
        P_img = resize_img(P_img, self.opt.loadSize, self.transform, itype=self.iuv_mode)
        AP0 = prepare_images(P_img, self.opt.fineSize)


        K_img = self.cords_to_map(self.get_pose(K_path))
        K_img = resize_img(K_img, self.opt.loadSize, self.transform, itype='multi')
        AK0 = prepare_images(K_img, self.opt.fineSize)

        return AI0,AP0,AK0
    '''
    def choose_subseq(self, paths):
        curr_seq = randint(0, len(paths) - 1)
        curr_idx = randint(0, len(paths[curr_seq]) - 1 - self.subseq_len)
        return curr_seq,curr_idx

    def start_fullseq(self):
        self.subseq_idx = 0

        self.currA_idx = 0
        self.currA_obj = self.A_objid[self.currA_seq]

        self.currB_idx = 0
        self.currB_obj = self.B_objid[self.currB_seq]

        self.currB_keys = self.centroids[self.currB_obj]
        self.currA_keys = self.centroids[self.currA_obj]

        self.subseq_len = len(self.A_paths[self.currA_seq])

    def start_new_subseq(self):
        self.subseq_idx = 0

        self.currA_seq, self.currA_idx = self.choose_subseq(self.A_paths)
        self.currA_obj = self.A_objid[self.currA_seq]

        self.currB_seq, self.currB_idx = self.choose_subseq(self.B_paths)
        self.currB_obj = self.B_objid[self.currB_seq]

        self.currB_keys = self.centroids[self.currB_obj]
        self.currA_keys = self.centroids[self.currA_obj]

    def get_current_keys(self):
        return self.currB_keys, self.currA_keys

    def load_packets(self,itemlist,ityep = 'img'):
        imglist = []
        for f in itemlist:
            AP_img = Image.open(f).convert('RGBA')  # SM - changed to RGBA (instead of RGB)
            AP_img = resize_img(AP_img, self.opt.loadSize, self.transform, ityep)
            AP0,_ = prepare_images(AP_img, self.opt.fineSize)
            AP0 = AP0.unsqueeze(0)
            imglist.append(AP0)
        packet = np.concatenate(imglist, axis=0)
        return packet

    def __getitem__(self, index):
        if self.subseq_idx == self.subseq_len :
            if not self.use_fullseq :
                self.start_new_subseq()
            else :
                self.currA_seq = (self.currA_seq + 1) % len(self.A_paths)
                self.currB_seq = (self.currB_seq + 1) % len(self.B_paths)
                self.start_fullseq()

        A_idx = self.currA_idx + self.subseq_idx % (len(self.A_paths[self.currA_seq]) - self.packet_len)
        B_idx = self.currB_idx + self.subseq_idx % (len(self.B_paths[self.currB_seq]) - self.packet_len)
        A_path = self.A_paths[self.currA_seq][A_idx:A_idx + self.packet_len]
        AP_path = self.AP_paths[self.currA_seq][A_idx :A_idx + self.packet_len]
        B_path = self.B_paths[self.currB_seq][B_idx:B_idx+self.packet_len]
        BP_path = self.BP_paths[self.currB_seq][B_idx:B_idx+self.packet_len]

        A0 = self.load_packets(A_path, ityep='img')
        B0 = self.load_packets(B_path, ityep='img')
        AP0 = self.load_packets(AP_path, ityep=self.iuv_mode)
        BP0 = self.load_packets(BP_path, ityep=self.iuv_mode)

        if self.subseq_idx == 0 :
            B_keys = self.currB_keys
            A_keys = self.currA_keys
        else :
            B_keys, A_keys = [], []

        self.subseq_idx += 1


        return {'A0': A0,'B0': B0,  'AP0': AP0, 'BP0': BP0,
                'A_paths': A_path, 'B_paths': B_path, 'imgidx' : self.subseq_idx ,
                'A_keys' : A_keys, 'B_keys' : B_keys, 'A_obj' : self.currA_obj,'B_obj' : self.currB_obj}

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'UnalignedSequenceDataset'


if __name__ == '__main__':
    pass