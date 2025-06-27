import math
import logging
import pdb
from functools import partial
from collections import OrderedDict
from copy import deepcopy
from abc import ABC, abstractmethod

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from ..layers.patch_embed import PatchEmbed
from ..utils import combine_tokens, recover_tokens, token2feature, feature2token
from ..vit.vit import VisionTransformer
from ..layers.attn_blocks import CEBlock, candidate_elimination_prompt

_logger = logging.getLogger(__name__)


class Fovea(nn.Module):

    def __init__(self, smooth=False):
        super().__init__()

        self.softmax = nn.Softmax(dim=-1)

        self.smooth = smooth
        if smooth:
            self.smooth = nn.Parameter(torch.zeros(1) + 10.0)

    def forward(self, x):
        '''
            x: [batch_size, features, k]
        '''
        b, c, h, w = x.shape
        x = x.contiguous().view(b, c, h*w)

        if self.smooth:
            mask = self.softmax(x * self.smooth)
        else:
            mask = self.softmax(x)
        output = mask * x
        output = output.contiguous().view(b, c, h, w)

        return output

class SCA(nn.Module):
    def __init__(self):
        super(SCA, self).__init__()
        self.squeeze_rgb = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_rgb = nn.Sequential(
            nn.Conv2d(32, 32, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid())

        self.squeeze_depth = nn.AdaptiveAvgPool2d(1)
        self.channel_attention_depth = nn.Sequential(
            nn.Conv2d(32, 32, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Sigmoid())

        self.cross_conv = nn.Conv2d(32*2, 32, 1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, x3_r, x3_d):
        
        SCA_ca = self.channel_attention_rgb(self.squeeze_rgb(x3_r))
        #print("SCA_ca:",SCA_ca.shape)
        SCA_3_o = x3_r * SCA_ca.expand_as(x3_r)
        #print("SCA_3_o:",SCA_3_o.shape)

        SCA_d_ca = self.channel_attention_depth(self.squeeze_depth(x3_d))
        #print("SCA_d_ca:",SCA_d_ca.shape)
        SCA_3d_o = x3_d * SCA_d_ca.expand_as(x3_d)
        #print("SCA_3d_o:",SCA_3d_o.shape)

        Co_ca3 = torch.softmax(SCA_ca + SCA_d_ca,dim=1)
        #print("Co_ca3:",Co_ca3.shape)

        SCA_3_co = x3_r * Co_ca3.expand_as(x3_r)
        #print("SCA_3_co:",SCA_3_co.shape)
        SCA_3d_co= x3_d * Co_ca3.expand_as(x3_d)
        #print("SCA_3d_co:",SCA_3d_co.shape)

        CR_fea3_rgb = SCA_3_o + SCA_3_co
        #print("CR_fea3_rgb:",CR_fea3_rgb.shape)
        CR_fea3_d = SCA_3d_o + SCA_3d_co

        CR_fea3 = torch.cat([CR_fea3_rgb,CR_fea3_d],dim=1)
        #print("CR_fea3:",CR_fea3.shape)
        CR_fea3 = self.cross_conv(CR_fea3)
        #print("CR_fea3:",CR_fea3.shape)

        return CR_fea3

class AFP(nn.Module):
    # This Module is the inner part of the AFP module described in the paper.
    # PaperAFP = ProMFTDeep.PromptBlock
    # PaperAFP has one AFP module with 3 additional convs, applied to input H, input P, and output P.
    def __init__(self):
        super(AFP, self).__init__()
        # self.channel_attention_rgb = nn.Sequential(
        #     nn.Conv2d(32, 32, 1, padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.Sigmoid())
        self.GAP_FC_rgb = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # works as global average pooling
            nn.Flatten(1),
            nn.Linear(32, 32),
            nn.Sigmoid()
        )

        # self.GAP_aux = nn.AdaptiveAvgPool2d(1)
        # self.channel_attention_aux = nn.Sequential(
        #     nn.Conv2d(32, 32, 1, padding=0),
        #     nn.ReLU(inplace=True),
        #     nn.Sigmoid())
        
        self.GAP_FC_aux = nn.Sequential(
            nn.AdaptiveAvgPool2d(1), # works as global average pooling
            nn.Flatten(1),
            nn.Linear(32, 32),
            nn.Sigmoid()
        )

        self.out_conv = nn.Conv2d(32*2, 32, 1, padding=0)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(std=0.01)
                m.bias.data.fill_(0)

    def forward(self, H, P):
        # Memo: Hl: Tensor[16, 32, 8, 8], $H_{l}$
        # Memo: Pl: Tensor[16, 32, 8, 8], $P_{l}$
        # possibly [B, C, H, W]
        # Memo: in the paper, equation(1) says $P_{l+1} = AFP(P_{l}, H_{l})$. But in the code we need to put $H_{l}$ at front.
        
        W_H = self.GAP_FC_rgb(H) # Memo: W_H: Tensor[16, 32]
        W_P = self.GAP_FC_aux(P) # Memo: W_P: Tensor[16, 32]
        W_F = F.softmax(W_H + W_P, dim=1) # Memo: W_F: Tensor[16, 32]
        
        H_CA_stride = (W_H + W_F).unsqueeze(-1).unsqueeze(-1).expand_as(H) * H # Memo: Tensor[16, 32, 8, 8]
        P_CA_stride = (W_P + W_F).unsqueeze(-1).unsqueeze(-1).expand_as(P) * P # Memo: Tensor[16, 32, 8, 8]

        P_next = self.out_conv(torch.cat([P_CA_stride, H_CA_stride], dim=1)) # Memo: P_next: Tensor[16, 32, 8, 8]
        # P_next: $P_{l+1}$
        
        # # OLD CODE:
        # SCA_ca = self.channel_attention_rgb(self.GAP_rgb(H))
        # SCA_3_o = H * SCA_ca.expand_as(H)

        # SCA_d_ca = self.channel_attention_aux(self.GAP_aux(P))
        # SCA_3d_o = P * SCA_d_ca.expand_as(P)

        # Co_ca3 = torch.softmax(SCA_ca + SCA_d_ca,dim=1)

        # SCA_3_co = H * Co_ca3.expand_as(H)
        # SCA_3d_co= P * Co_ca3.expand_as(P)

        # CR_fea3_rgb = SCA_3_o + SCA_3_co
        # CR_fea3_aux = SCA_3d_o + SCA_3d_co

        # P_next = self.cross_conv(
        #             torch.cat([CR_fea3_rgb,CR_fea3_aux],dim=1))
        # # P_next: $P_{l+1}$

        return P_next
        # Memo: Pl_plus: Tensor[16, 32, 8, 8]

class SKConv(nn.Module):
    def __init__(self, features):
        """ Constructor
        Args:
            features: input channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(SKConv, self).__init__()
        #d = max(int(features/r), L)
        d = 32
        self.M = 2
        self.features = features
        self.convs = nn.ModuleList([])
        for i in range(2):
            self.convs.append(nn.Sequential(
                nn.Conv2d(features, features, kernel_size=3+i*2, stride=1, padding=1+i),
                nn.BatchNorm2d(features),
                nn.ReLU(inplace=False)
            ))
        # self.gap = nn.AvgPool2d(int(WH/stride))
        self.fc = nn.Linear(features, d)
        self.fcs = nn.ModuleList([])
        for i in range(2):
            self.fcs.append(
                nn.Linear(d, features)
            )
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x1, x2):
        
        x1 = x1.unsqueeze_(dim=1)
        x2 = x2.unsqueeze_(dim=1)
            
        feas = torch.cat([x1, x2], dim=1)
        fea_U = torch.sum(feas, dim=1)
        # fea_s = self.gap(fea_U).squeeze_()
        fea_s = fea_U.mean(-1).mean(-1)
        fea_z = self.fc(fea_s)
        for i, fc in enumerate(self.fcs):
            vector = fc(fea_z).unsqueeze_(dim=1)
            if i == 0:
                attention_vectors = vector
            else:
                attention_vectors = torch.cat([attention_vectors, vector], dim=1)
        attention_vectors = self.softmax(attention_vectors)
        attention_vectors = attention_vectors.unsqueeze(-1).unsqueeze(-1)
        fea_v = (feas * attention_vectors).sum(dim=1)
        return fea_v


class MCG(nn.Module):
    def __init__(self, rgb_inchannels, depth_inchannels):
        super(MCG, self).__init__()
        self.channels = rgb_inchannels
        self.convDtoR = nn.Conv2d(depth_inchannels, rgb_inchannels, 3,1,1)
        self.convTo2 = nn.Conv2d(rgb_inchannels*2, 2, 3, 1, 1)
        self.sig = nn.Sigmoid()
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.global_max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(self.channels, self.channels // 16, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(self.channels // 16, self.channels, 1, bias=False)

        self.MlpConv = nn.Conv2d(self.channels, self.channels, 1, bias=False)
        

    def forward(self,r,d):
        d = self.convDtoR(d)
        d = self.relu(d)
        H = torch.cat((r,d), dim=1)
        H_conv = self.convTo2(H)
        H_conv = self.sig(H_conv)
        g = self.global_avg_pool(H_conv)

        ga = g[:, 0:1, :, :]
        gm = g[:, 1:, :, :]

        Ga = r * ga
        Gm = d * gm

        '''
        GmA = self.global_avg_pool(Gm)

        GmA_fc = self.fc2(self.relu(self.fc1(GmA)))
        GmA_fc = self.sig(GmA_fc)
        Gm1 = Gm * GmA_fc

        Gm1M = self.global_max_pool(Gm1)
        Gm1M_conv = self.MlpConv(Gm1M)
        Gm2 = self.sig(Gm1M_conv)

        Gm_out = Gm1 * Gm2
        # Gm_out = self.coordAttention(Gm)
        '''
        out = Gm + Ga

        return out


class BaseProMFT(VisionTransformer, ABC):
    """ Vision Transformer with candidate elimination (CE) module and prompt blocks.

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer_cls=None,
                 act_layer_cls=None, weight_init='', ce_loc=None, ce_keep_ratio=None, search_size=None, template_size=None,
                 new_patch_size=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
            new_patch_size: backbone stride
        """
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        if not norm_layer_cls: norm_layer_cls = partial(nn.LayerNorm, eps=1e-6)
        self.norm_layer_cls = norm_layer_cls
        act_layer_cls = act_layer_cls or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed_prompt = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim)) # it's redundant
        self.pos_drop = nn.Dropout(p=drop_rate)

        '''
        prompt parameters
        '''
        self.depth = depth
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_search=new_P_H * new_P_W
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_template=new_P_H * new_P_W
        """add here, no need use backbone.finetune_track """
        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))

        self.prompt_blocks = self.create_prompt_blocks()
        self.prompt_norms = self.create_prompt_norms()

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        blocks = []
        ce_index = 0
        self.ce_loc = ce_loc
        for i in range(depth):
            ce_keep_ratio_i = 1.0
            if ce_loc is not None and i in ce_loc:
                ce_keep_ratio_i = ce_keep_ratio[ce_index]
                ce_index += 1

            blocks.append(
                CEBlock(
                    dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                    attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer_cls, act_layer=act_layer_cls,
                    keep_ratio_search=ce_keep_ratio_i)
            )

        self.blocks = nn.Sequential(*blocks)
        self.norm = norm_layer_cls(embed_dim)

        self.init_weights(weight_init)

    @abstractmethod
    def create_prompt_blocks(self) -> nn.Sequential:
        raise NotImplementedError()
    
    @abstractmethod
    def create_prompt_norms(self) -> nn.Sequential:
        raise NotImplementedError()

    # override
    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         ce_template_mask=None, ce_keep_rate=None,
                         return_last_attn=False):
        # MEMO:
        # z: Tensor[16, 6, 128, 128]
        # x: Tensor[16, 6, 256, 256]
        # ce_template_mask: Tensor[16, 64] (dtype=boolean)
        # ce_keep_rate: 1

        B, C, H, W = x.shape
        #print("model input x:",x.shape)
        #print("model input z:",z.shape)
        # rgb_img
        x_rgb = x[:, :3, :, :]
        z_rgb = z[:, :3, :, :]
        # depth thermal event images
        x_dte = x[:, 3:, :, :]
        z_dte = z[:, 3:, :, :]
        # overwrite x & z
        x, z = x_rgb, z_rgb

        z = self.patch_embed(z) # z: Tensor[16, 64, 768]
        x = self.patch_embed(x) # x: Tensor[16, 256, 768]
        # print("patch_embed x:",x.shape)
        # print("patch_embed z:",z.shape)

        z_dte = self.patch_embed_prompt(z_dte) # Tensor[16, 64, 768]
        x_dte = self.patch_embed_prompt(x_dte) # Tensor[16, 256, 768]

        '''input prompt: by adding to rgb tokens'''
        z, x = self.forward_block_0(z, x, z_dte, x_dte) # z, x: Tensor[16, 64, 768], Tensor[16, 256, 768]

        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z += self.pos_embed_z # self.pos_embed_z: Tensor[1, 64, 768]
        x += self.pos_embed_x # self.pos_embed_x: Tensor[1, 256, 768]

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x = combine_tokens(z, x, mode=self.cat_mode) # x: Tensor[16, 320, 768]. (self.cat_mode: 'direct')
        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)

        x = self.pos_drop(x) # x: Tensor[16, 320, 768]

        lens_z = self.pos_embed_z.shape[1] # lens_z = 64
        lens_x = self.pos_embed_x.shape[1] # lens_x = 256

        global_index_t = torch.linspace(0, lens_z - 1, lens_z, dtype=torch.int64).to(x.device)
        global_index_t = global_index_t.repeat(B, 1) # Tensor[16, 64]

        global_index_s = torch.linspace(0, lens_x - 1, lens_x, dtype=torch.int64).to(x.device)
        global_index_s = global_index_s.repeat(B, 1) # Tensor[16, 256]

        removed_indexes_s = []
        removed_flag = False
        for i, blk in enumerate(self.blocks):
            '''
            add parameters prompt from 1th layer
            '''
            if i >= 1:
                z, x = self.forward_block_i(x=x, z=z, x_dte=x_dte, z_dte=z_dte, i=i, batch_size=B, lens_x=lens_x, lens_z=lens_z, global_index_t=global_index_t, global_index_s=global_index_s, removed_indexes_s=removed_indexes_s)
            #print("block input:",x.shape)
            x, global_index_t, global_index_s, removed_index_s, attn = \
                blk(x, global_index_t, global_index_s, mask_x, ce_template_mask, ce_keep_rate)
            # ^ x: Tensor[16, 320, 768]

            if self.ce_loc is not None and i in self.ce_loc:
                removed_indexes_s.append(removed_index_s)

        x = self.norm(x)
        lens_x_new = global_index_s.shape[1]
        lens_z_new = global_index_t.shape[1]

        z = x[:, :lens_z_new]
        x = x[:, lens_z_new:]

        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)

            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            # recover original token order
            C = x.shape[-1]
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)

        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode)

        # re-concatenate with the template, which may be further used by other modules
        x = torch.cat([z, x], dim=1)

        aux_dict = {
            "attn": attn,
            "removed_indexes_s": removed_indexes_s,  # used for visualization
        }

        return x, aux_dict

    # override
    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False):
        # MEMO: 
        # z: Tensor[16, 6, 128, 128]
        # x: Tensor[16, 6, 256, 256]

        x, aux_dict = self.forward_features(z, x, ce_template_mask=ce_template_mask, ce_keep_rate=ce_keep_rate,)
        # x: ??
        # aux_dict: ??

        return x, aux_dict

    def forward_block_0(self, z, x, z_dte, x_dte):
        z = z + z_dte
        x = x + x_dte
        return z, x
    
    def forward_block_i(self, x, z, x_dte, z_dte, i, batch_size, lens_x, lens_z, global_index_t, global_index_s, removed_indexes_s):
        return z, x

    @classmethod
    def build(cls, pretrained=False, **kwargs): # TODO: use cfg
        model = cls(**kwargs)

        if pretrained:
            if 'npz' in pretrained:
                model.load_pretrained(pretrained, prefix='')
            else:
                checkpoint = torch.load(pretrained, map_location="cpu")
                missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
                print('Load pretrained OSTrack from: ' + pretrained)
                print(f"missing_keys: {missing_keys}")
                print(f"unexpected_keys: {unexpected_keys}")

        return model

    @classmethod
    def build_base(cls, pretrained=False, **kwargs):
        """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
        """
        model_kwargs = dict(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
        model = cls.build(pretrained=pretrained, **model_kwargs)
        return model

    @classmethod
    def build_large(cls, pretrained=False, **kwargs):
        """ ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
        """
        model_kwargs = dict(
            patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
        model = cls.build(pretrained=pretrained, **model_kwargs)
        return model
