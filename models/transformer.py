import copy
import torch
import torch.nn as nn
from .linear_attention import LinearAttention, FullAttention


class EncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(EncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message



class DecoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(DecoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention() if attention == 'linear' else FullAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model*2, d_model*2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model*2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead*self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message

    def get_attention_map(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        '''
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        '''
        query = self.q_proj(query).view(bs, query.shape[1], 1, -1)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, key.shape[1], 1, -1)  # [N, L, (H, D)]

        QK = torch.einsum("nlhd,nshd->nlsh", query, key)

        # Compute the attention and the weighted average
        softmax_temp = 1. / query.size(3) ** .5  # sqrt(D)
        A = torch.softmax(softmax_temp * QK, dim=2)

        return A[0,-1:,:,0]

class Transformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, d_model = 128, nhead=2, attention = 'linear',use_q_self_attention = False):
        super(Transformer, self).__init__()
        self.d_model = d_model
        self.num_encoder_layers = 2
        self.num_decoder_layers = 2
        self.use_q_self_attention = use_q_self_attention

        if not self.use_q_self_attention :
            self.decoder_layer_names = ['cross' for _ in range(self.num_decoder_layers)]
        else :
            self.decoder_layer_names = sum([['self','cross'] for _ in range(self.num_decoder_layers)],[])

        encoder_layer = EncoderLayer(d_model = d_model, nhead=nhead, attention = attention)
        decoder_layer = DecoderLayer(d_model=d_model, nhead=nhead, attention=attention)
        self.encoder_layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(self.num_encoder_layers)])
        self.decoder_layers = nn.ModuleList([copy.deepcopy(decoder_layer) for _ in range(len(self.decoder_layer_names))])
        self._reset_parameters()
        self.set_debug_mode(False)
        self.debug = [[],[]]

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def set_debug_mode(self,mode):
        self.gDEBUG_MODE = mode

    def get_debug(self):
        import numpy as np
        debug = copy.deepcopy(self.debug)
        self.debug = [[], []]
        try :
            encoder_map0 = np.concatenate(debug[0], axis=0).transpose()
            encoder_map1 = np.concatenate(debug[1], axis=0).transpose()
        except :
            encoder_map0,encoder_map1 = [],[]

        return encoder_map0,encoder_map1

    def forward(self, keys, feat, mask_keys=None, mask_feat=None):
        """
        Args:
            keys (torch.Tensor): [N, L, C]
            feat (torch.Tensor): [N, S, C]
            mask_keys (torch.Tensor): [N, L] (optional)
            mask_feat (torch.Tensor): [N, S] (optional)

        (N - batch size , L/S  - num_samples, C - feature vector size)
        """

        assert self.d_model == feat.size(2), "the feature number of src and transformer must be equal"

        for layer in self.encoder_layers:
            mems = layer(keys, keys, mask_keys, mask_keys)

        count = 0
        for layer, name in zip(self.decoder_layers, self.decoder_layer_names):
            if name == 'self':
                feat = layer(feat, feat, mask_feat, mask_feat)
            elif name == 'cross':
                feat = layer(feat, mems, mask_feat, mask_keys)
                if self.gDEBUG_MODE : #SM DEBUG - creating attention maps
                    A = layer.get_attention_map(feat, mems)
                    self.debug[count].append(A.cpu().detach().numpy())
                    count = count + 1
            else:
                raise KeyError
        return feat, mems

'''
for layer in self.decoder_layers:
    feat = layer(feat, feat, mask_feat, mask_keys) if self.use_q_self_attention else feat
    feat = layer(feat, mems, mask_feat, mask_keys)
'''