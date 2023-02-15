import os.path
import torchvision.transforms as transforms
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random
import torch

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np

from common.fileutils import get_dirs_list
import json
import yaml

from common.fileutils import fileparts
from torchvision import models
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pickle as pk


from data.unaligned_person_dataset import _makepath,prepare_images,resize_img

MISSING_VALUE = -1
from torch.utils.data import Dataset

class CasiaADataset(Dataset):
    def __init__(self, dataparams,obj_id=None,loadSize=256,fineSize=256,iuv_mode='iuv1'):
        self.dataparams = dataparams
        self.obj_id = obj_id
        self.remove_target_from_source = True
        self.load_keypoints = True
        self.iuv_mode = iuv_mode
        self.serial_batches = False
        self.loadSize = loadSize
        self.fineSize = fineSize
        self.use_cuda = True
        self.device = torch.device("cuda" if self.use_cuda else "cpu")
        self.use_filters = True

        with open(self.dataparams, 'r') as stream:
            pdata = yaml.safe_load(stream)
            self.pdata = pdata['DATASET']

        self.dataroot = self.pdata['DATA_ROOT']
        self.filters = self.pdata['FILTERS'] if self.use_filters else None

        sobj = [self.obj_id]

        self.SI_paths = _makepath(self.dataroot, sobj, itype=self.pdata['RGB_DIR'],filters=self.filters)
        self.SP_paths = _makepath(self.dataroot, sobj, itype=self.pdata['IUV_DIR'],filters=self.filters)
        self.SK_paths = _makepath(self.dataroot, sobj, itype=self.pdata['KPS_DIR'],filters=self.filters, images_only=False)

        self.dir_SI = os.path.join(self.dataroot,  self.pdata['RGB_DIR']) #person images
        self.dir_SK = os.path.join(self.dataroot,  self.pdata['KPS_DIR']) #keypoints
        self.dir_SP = os.path.join(self.dataroot,  self.pdata['IUV_DIR']) #edge


        self.SK_paths = sorted(self.SK_paths)
        self.SI_paths = sorted(self.SI_paths)
        self.SP_paths = sorted(self.SP_paths)

        self.SI_size = len(self.SI_paths)

        transform_list = [transforms.ToTensor(),
                          transforms.Normalize((0.5, 0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5, 0.5))] #SM added layer for alpha
        self.transform = transforms.Compose(transform_list)

        #self.centroids = self.get_centroids(self.obj_id)

    '''
    def get_features(self,P):
        features = self.features_model(P[0:3, :, :].unsqueeze(0).to(self.device))
        x = self.pca.transform(features.cpu().detach().numpy())
        return x

    #RI, RP, RK = self.get_close_centroid(BP0)

    def get_close_centroid(self,P):
        center = self.get_features(P)
        v = [sum((lf.squeeze(0) - center.squeeze(0)) ** 2) for lf in self.CF]
        amin = np.argmin(v)
        RI,RP,RK = self.CI[amin],self.CP[amin],self.CK[amin]
        return RI,RP,RK

    def get_centroids(self, obj_id):
        dir_SI = os.path.join(self.dataroot, self.pdata['RGB_DIR'])  # person images
        dir_SK = os.path.join(self.dataroot, self.pdata['KPS_DIR'])  # keypoints
        dir_SP = os.path.join(self.dataroot, self.pdata['IUV_DIR'])  # edge
        dir_centroids = os.path.join(self.dataroot, 'centroids')  # edge
        pcentroid = self.pdata['CENTROIDS'][obj_id]

        self.features_model = models.vgg16_bn(pretrained=True).to(self.device)

        # later reload the pickle file
        self.pca = pk.load(open(os.path.join(dir_centroids,f"{obj_id}_PCA.pkl"), 'rb'))

        self.CI,self.CP,self.CK,self.CF =[],[],[],[]
        for item in pcentroid :
            K_path = os.path.join(dir_SK, obj_id, item)
            p, f, e = fileparts(K_path.replace(dir_SI,dir_SK))
            K_path =  os.path.join(p, f'{f}_keypoints.json')

            I,P,K = self.get_data(os.path.join(dir_SI, obj_id, item), os.path.join(dir_SP, obj_id, item), K_path)
            x = self.get_features(P)

            self.CI.append(I)
            self.CP.append(P)
            self.CK.append(K)
            self.CF.append(x)
    '''
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

    def get_data(self,I_path,P_path,K_path):
        img = Image.open(I_path).convert('RGBA')  #SM - changed to RGBA (instead of RGB)
        iuv = Image.open(P_path).convert('RGBA')  # SM - changed to RGBA (instead of RGB)

        img = resize_img(img, self.loadSize, self.transform, itype='img')
        A0,_ = prepare_images(img, self.fineSize)

        iuv = resize_img(iuv, self.loadSize, self.transform, itype=self.iuv_mode)
        AP0,_ = prepare_images(iuv, self.fineSize)

        if self.load_keypoints :
            hms = self.cords_to_map(self.get_pose(K_path))
            hms = resize_img(hms, self.loadSize, self.transform, itype='multi')
            AK0,_ = prepare_images(hms, self.fineSize)
        else :
            AK0 =[]

        return A0,AP0,AK0

    def __getitem__(self, index):
        index_2 = index % self.SI_size

        # At inference time we try to convert pose2 (source) to appearance of img1 (target)
        img1, iuv1, pose1 = self.get_data(self.SI_paths[index_2], self.SP_paths[index_2], self.SK_paths[index_2])

        return {'img1': img1,'iuv1': iuv1,  'pose1': pose1,
                'paths': self.SI_paths[index_2] , 'idx':index_2}

    def get_frames(self, rgb_path):
        kps_path = rgb_path.replace(self.dir_SI, self.dir_SK)
        iuv_path = rgb_path.replace(self.dir_SI, self.dir_SP)

        # At inference time we try to convert pose2 (source) to appearance of img1 (target)
        img1, iuv1, pose1 = self.get_data(rgb_path, iuv_path, kps_path)

        return {'img1': img1,'iuv1': iuv1,  'pose1': pose1,
                'paths': rgb_path , 'idx':-1}

    def __len__(self):
        return max(0, self.SI_size)

    def name(self):
        return 'CasiaADataset'


