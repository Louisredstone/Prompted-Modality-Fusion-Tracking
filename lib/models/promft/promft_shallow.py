import torch
from torch import nn
from .base_promft import BaseProMFT
from ..layers.patch_embed import PatchEmbed
from .base_promft import SCA
from .promft_deep import PromptBlock
from ..utils import combine_tokens, recover_tokens, token2feature, feature2token

class ProMFTShallow(BaseProMFT):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer_cls=None,
                 act_layer_cls=None, weight_init='', ce_loc=None, ce_keep_ratio=None, search_size=None, template_size=None,
                 new_patch_size=None,):
        super(ProMFTShallow, self).__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, representation_size=representation_size, distilled=distilled, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, embed_layer=embed_layer, norm_layer_cls=norm_layer_cls, act_layer_cls=act_layer_cls, weight_init=weight_init, ce_loc=ce_loc, ce_keep_ratio=ce_keep_ratio, search_size=search_size, template_size=template_size, new_patch_size=new_patch_size)
        
    def create_prompt_blocks(self):
        return nn.Sequential(
            PromptBlock(inplanes=self.embed_dim, hide_channel=32, smooth=True)
        )
    
    def create_prompt_norms(self):
        return nn.Sequential(
            self.norm_layer_cls(self.embed_dim)
        )
        
    def forward_block_0(self, z, x, z_dte, x_dte):
        z_feat = token2feature(self.prompt_norms[0](z))
        x_feat = token2feature(self.prompt_norms[0](x))
        z_dte_feat = token2feature(self.prompt_norms[0](z_dte))
        x_dte_feat = token2feature(self.prompt_norms[0](x_dte))
        z_feat = torch.cat([z_feat, z_dte_feat], dim=1)
        x_feat = torch.cat([x_feat, x_dte_feat], dim=1)
        #print("prompt block input:",x_feat.shape,z_feat.shape)
        z_feat = self.prompt_blocks[0](z_feat) # after, will call self.prompt_blocks[i]
        x_feat = self.prompt_blocks[0](x_feat)
        z_dte = feature2token(z_feat)
        x_dte = feature2token(x_feat)

        z = z + z_dte
        x = x + x_dte
        
        return z, x