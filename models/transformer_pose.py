import torch
from common.imutils import imshowpts

import torch.nn as nn
from models.transformer import Transformer


class PoseTransformer(nn.Module):
    def __init__(self, d_model, q_size = 10,k_size = 20,use_q_self_attention = False): #, qseq_max_len = 100
        super(PoseTransformer, self).__init__()
        self.pool = nn.MaxPool2d(4, stride=4, return_indices=True)
        self.unpool = nn.MaxUnpool2d(4, stride=4)

        self.QPos = nn.Parameter(torch.randn(1,q_size , d_model))
        self.KPos = nn.Parameter(torch.randn(1,k_size, d_model))

        in_dim = d_model // 4

        self.transformer = Transformer(d_model=in_dim,use_q_self_attention=use_q_self_attention)
        self.to_q = nn.Linear(d_model, in_dim, bias=False)
        self.to_k = nn.Linear(d_model, in_dim, bias=False)
        self.from_q = nn.Linear(in_dim, d_model, bias=False)
        self.from_k = nn.Linear(in_dim, d_model, bias=False)

        self.mask_keys = None
        self.mask_feat = None
        self.debug_mode = False
        self.count = 0

    def rearange(self,X):
        X = X.unsqueeze(0)
        b,n,c,w,h = X.size()
        X = X.view(b,n,-1)
        return X

    def inverse(self,X,sz):
        X = X.squeeze(0)
        X = X.view(X.size(0),-1,sz[2],sz[3])
        return X
    def start_new_sequence(self):
        self.count = 0

    def set_debug_mode(self,mode):
        self.debug_mode = True
        self.transformer.set_debug_mode(mode)

    def get_debug(self):
        encoder_map0, encoder_map1 = self.transformer.get_debug()
        return encoder_map0, encoder_map1

    def forward(self, K,Q ) :
        #if self.debug_mode :
        #   self.mask_keys = torch.zeros(1,K.shape[0]).to(K.device)
        #    self.mask_feat = torch.zeros(1, Q.shape[0]).to(Q.device)
            #self.mask_keys[:,4] = 1.0

        K,_ = self.pool(K)
        Q,indices  =  self.pool(Q)
        szK,szQ = K.size(), Q.size()

        K,Q = self.rearange(K) , self.rearange(Q)

        Q = self.to_q(Q)
        K = self.to_k(K)

        Q,mems = self.transformer(keys=K,feat =Q , mask_keys = self.mask_keys , mask_feat = self.mask_feat)

        Q = self.from_q(Q)

        outQ = self.inverse(Q, szQ)
        Q = self.unpool(outQ, indices)

        return Q  # ,mems

        '''
        self.KMask[:,0:K.shape[1]] = 1
        self.QMask[:, 1:] = self.QMask[:, 0:-1].clone()
        self.QMask[:, 0] = 1
        '''
