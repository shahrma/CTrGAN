import os.path
from data.image_folder import make_dataset
from PIL import Image
import random
import torch

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import numpy as np
from common.imutils import imshowpts
from common.fileutils import get_dirs_list

def filter_filelist(fileslist, filters) :
    if filters is None :
        return fileslist
    out = []
    for f in fileslist :
        sf = os.path.dirname(f).split(os.sep)[-len(filters):]
        check = True
        for idx,_ in enumerate(filters) :
            if sf[idx] not in filters[idx] :
                check = False

        if check :
            out.append(f)

    return sorted(out)


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

def makeseq(dataroot, objs, itype = None,images_only=True,filters=None) :
    paths = []
    objid = []
    for item in objs :
        dirslist = get_dirs_list(os.path.join(dataroot,itype,item))
        if filters is not None :
            dirslist = filter_dirs(os.path.join(dataroot, itype, item), dirslist, filters=filters)
        for currdir in dirslist:
            paths.append(sorted(make_dataset(currdir,images_only=images_only)))
            objid.append(item)
    return paths,objid
'''
def prepare_images(A_img, fineSize):
    w_total = A_img.size(2)
    w = int(w_total)
    h = A_img.size(1)
    w_offset = random.randint(0, max(0, w - fineSize - 1))
    h_offset = random.randint(0, max(0, h - fineSize - 1))

    A0 = A_img[:, h_offset:h_offset + fineSize,
         w_offset:w_offset + fineSize]

    return A0
'''

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
    '''
    elif itype == 'iuv1' :
        imgB = img.resize((loadSize, loadSize), Image.BICUBIC)
        imgN = img.resize((loadSize, loadSize), Image.NEAREST)
        img = torch.tensor(np.array(imgB)) * (2 / 255) - 1
        imgN = torch.tensor(np.array(imgN)*10) * (2 / 255) - 1
        img[:, :, 2] = imgN[:, :, 2]
        img = img.permute(2, 0, 1)
    '''

    return img


import torchvision.transforms as transforms

def tensor2iuv2(img_):
    img_ = img_.squeeze(0)
    img_ = img_.permute(1,2,0)

    luv = (img_[:, :, 0:2]  + 1)  * 255 / 2
    li = img_[:, :, 2]
    la = (img_[:, :, 3]  + 1)  * 255 / 2

    li = ( ((li + 1)*255/2) / 10)

    li = li.unsqueeze(2) if  len(li.shape) < 3 else li
    la = la.unsqueeze(2) if len(la.shape) < 3 else la

    img = torch.cat((luv, li, la), axis=2)
    img = torch.clamp(img,min=0,max=255)
    img = img.numpy().astype(np.uint8)

    return img
def load_centroids_images(itemlist, iuv_mode ='iuv1'):
    transform_list = [transforms.ToTensor(),
                      transforms.Normalize((0.5, 0.5, 0.5, 0.5),
                                           (0.5, 0.5, 0.5, 0.5))]  # SM added layer for alpha
    transform = transforms.Compose(transform_list)


    imglist = []
    for f in itemlist:
        AP_img = Image.open(f).convert('RGBA')  # SM - changed to RGBA (instead of RGB)
        loadSize = fineSize = AP_img.size[0]
        AP_img = resize_img(AP_img, loadSize, transform, itype=iuv_mode)
        AP0,_ = prepare_images(AP_img, fineSize)
        AP0 = AP0.unsqueeze(0)
        imglist.append(AP0)

    centroids = np.concatenate(imglist, axis=0)

    return centroids

def get_centroids_path(dataroot,centroids_path , item ,filters = None,itype = 'iuv'):
    filelist = [os.path.join(dataroot, itype, item, file) for file in centroids_path[item]]
    if filters is not None:
        itemlist = filter_dirs(os.path.join(dataroot, itype, item), filelist, filters=filters)
    else:
        itemlist = filelist

    return itemlist

def get_dataparams(yamlfile) :
    import yaml

    with open(yamlfile, 'r') as stream:
        pdata = yaml.safe_load(stream)
        pdata = pdata['DATASET']
    return pdata