from torch import nn
from .base_promft import BaseProMFT
from ..layers.patch_embed import PatchEmbed
from .base_promft import SCA

class PromptBlockNaive(nn.Module,):
    def __init__(self, inplanes=None, hide_channel=None, smooth=False):
        super(PromptBlockNaive, self).__init__()

    def forward(self, x):
        return x

class ProMFTNaive(BaseProMFT):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer_cls=None,
                 act_layer_cls=None, weight_init='', ce_loc=None, ce_keep_ratio=None, search_size=None, template_size=None,
                 new_patch_size=None,):
        super(ProMFTNaive, self).__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, representation_size=representation_size, distilled=distilled, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, embed_layer=embed_layer, norm_layer_cls=norm_layer_cls, act_layer_cls=act_layer_cls, weight_init=weight_init, ce_loc=ce_loc, ce_keep_ratio=ce_keep_ratio, search_size=search_size, template_size=template_size, new_patch_size=new_patch_size)
        
    def create_prompt_blocks(self):
        blocks = [PromptBlockNaive(inplanes=self.embed_dim, hide_channel=32, smooth=True)
            for i in range(self.depth)]
        return nn.Sequential(*blocks)
    
    def create_prompt_norms(self):
        norms = [nn.LayerNorm(self.embed_dim) for i in range(self.depth)]
        return nn.Sequential(*norms)
    
class ProMFTNaiveShallow(ProMFTNaive):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=1, # <- depth = 1.
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer_cls=None,
                 act_layer_cls=None, weight_init='', ce_loc=None, ce_keep_ratio=None, search_size=None, template_size=None,
                 new_patch_size=None,):
        super(ProMFTNaiveShallow, self).__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, representation_size=representation_size, distilled=distilled, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, embed_layer=embed_layer, norm_layer_cls=norm_layer_cls, act_layer_cls=act_layer_cls, weight_init=weight_init, ce_loc=ce_loc, ce_keep_ratio=ce_keep_ratio, search_size=search_size, template_size=template_size, new_patch_size=new_patch_size)