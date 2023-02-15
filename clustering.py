
import os
from data.CasiaADataset import CasiaADataset
from torch.utils.data import DataLoader
import numpy as np
import json
from torchvision import models
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


def make_dataloader_train(target_id='0001',BATCHSIZE=2,NUM_WORKERS=0):
    train_set = CasiaADataset(dataparams='./data/CASIA-A_DS_MPS01.yaml',  obj_id=target_id,loadSize=224,fineSize=224,iuv_mode='img')

    train_loader = DataLoader(train_set,
        batch_size=BATCHSIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        drop_last = True
    )
    return train_loader

from common.fileutils import fileparts
def split_by_path(paths,filters = [['00'],['000','001','002','003']]) :
    p = [os.path.dirname(f) for f in paths]
    filtered = []
    for idx,pf in enumerate(p) :
        x = pf.split('/')
        status = True
        for idx_filter in range(len(filters)) :
            if x[-len(filters)+idx_filter] not in filters[idx_filter] :
                status = False
        if status :
            filtered.append(idx)

    return filtered

def choose_centroids(x,paths,n_clusters) :
    kmeans = KMeans(n_clusters=n_clusters)
    kmeans.fit(x)

    # holds the cluster id and the images { id: [images] }
    labels  = kmeans.labels_
    centers = kmeans.cluster_centers_
    cen_paths = []
    for idx in range(0,n_clusters) :
        lpath = [paths[i] for i,l in enumerate(labels) if l == idx]
        lfeat = x[labels==idx]
        center = centers[idx]
        amin = np.argmin([sum((lf-center)**2) for lf in lfeat])
        cen_paths.append(lpath[amin])
    return cen_paths
def train_stage(target_id,centroids_path,n_clusters = 24) :
    os.makedirs(centroids_path,exist_ok=True)
    device = 'cuda'

    data_loader = make_dataloader_train(target_id=target_id,BATCHSIZE=1)
    dataset_size = len(data_loader.dataset)
    print('#training images = %d' % dataset_size)
    model = models.vgg16_bn(pretrained=True).to(device)
    features=[]
    paths = []
    for bidx, inputs in enumerate(data_loader):
        data = inputs['iuv1'][:, 0:3, :, :].to(device)
        f = model(data)
        features.append(f[0].cpu().detach().numpy())
        paths.append(inputs['paths'][0])
        print(f"{bidx} - {inputs['paths'][0]}")

    pca = PCA(n_components=100, random_state=22)
    pca.fit(features)
    x = pca.transform(features)

    import pickle as pk
    pk.dump(pca, open(os.path.join(centroids_path,f"{target_id}_PCA.pkl"), "wb"))
    # cluster feature vectors
    views = ['00','45','90']
    cen_paths = []
    for view in views  :
        curridx = split_by_path(paths, filters=[[view], [ '001', '002']])
        paths_ = [paths[i] for i in curridx]
        x_  = x[curridx,:]
        cen_paths += choose_centroids(x_, paths_, round(n_clusters/len(views)))

    cen_paths = sorted(cen_paths)
    data = {}
    data['cen_paths'] = cen_paths
    with open(os.path.join(centroids_path,f'{target_id}-centroids.json'), 'w') as outfile:
        json.dump(data, outfile)

import yaml

def create_yaml_centroid(centroids_path,TARGET_LIST,prefix_path) :
    dict_file = {}
    dict_file['DATASET'] = {}
    dict_file['DATASET']['CENTROIDS'] = {}
    for target_id in TARGET_LIST :
        with open(os.path.join(centroids_path, f'{target_id}-centroids.json'), 'r') as outfile:
            data = json.load( outfile)
            data_t = [item.replace(os.path.join(prefix_path,f'{target_id}/'),'') for item in data['cen_paths']]
            dict_file['DATASET']['CENTROIDS'][target_id] = data_t

    return dict_file


if __name__ == '__main__':
    #n_clusters = 24
    n_clusters = 16
    n_clusters = 18
    #n_clusters = 64

    centroids_path = f'./centroids/centroids_{n_clusters:04}'
    prefix_path = '../../datasets/CASIA/DatasetA/Crop256/rgb/'

    TARGET_LIST = [f'{i:04}' for i in range(20)]
    for target_id in TARGET_LIST:
        train_stage(target_id=target_id,centroids_path=centroids_path,n_clusters=n_clusters)

    dict_file = create_yaml_centroid(centroids_path,TARGET_LIST,prefix_path)

    with open(os.path.join(centroids_path,f'centroids_{n_clusters}.yaml'), 'w') as file:
        documents = yaml.dump(dict_file, file)