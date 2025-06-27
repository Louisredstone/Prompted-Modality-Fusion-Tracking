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

from .layers.patch_embed import PatchEmbed
from .utils import combine_tokens, recover_tokens, token2feature, feature2token
from .vit.vit import VisionTransformer
from .layers.attn_blocks import CEBlock, candidate_elimination_prompt

_logger = logging.getLogger(__name__)

class NoAUXWrapper(VisionTransformer):
    '''
    A wrapper that wraps a VisionTransformer as something like a ProMFT class.
    NOTICE: AUX channel is required as an input, but designed to be unused in this wrapper.
    '''
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
        raise NotImplementedError()
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

        # self.prompt_blocks = self.create_prompt_blocks()
        # self.prompt_norms = self.create_prompt_norms()

        self.ce_loc = ce_loc

        self.norm = norm_layer_cls(embed_dim)

        self.init_weights(weight_init)

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

        # z_dte = self.patch_embed_prompt(z_dte) # Tensor[16, 64, 768]
        # x_dte = self.patch_embed_prompt(x_dte) # Tensor[16, 256, 768]

        '''input prompt: by adding to rgb tokens'''
        # z, x = self.forward_block_0(z, x, z_dte, x_dte) # z, x: Tensor[16, 64, 768], Tensor[16, 256, 768]

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
            # if i >= 1:
            #     z, x = self.forward_block_i(x=x, z=z, x_dte=x_dte, z_dte=z_dte, i=i, batch_size=B, lens_x=lens_x, lens_z=lens_z, global_index_t=global_index_t, global_index_s=global_index_s, removed_indexes_s=removed_indexes_s)
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
        } # unused. just a placeholder for compatibility with ProMFT.

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

    # def forward_block_0(self, z, x, z_dte, x_dte):
    #     # z = z + z_dte
    #     # x = x + x_dte
    #     return z, x
    
    # def forward_block_i(self, x, z, x_dte, z_dte, i, batch_size, lens_x, lens_z, global_index_t, global_index_s, removed_indexes_s):
    #     return z, x

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