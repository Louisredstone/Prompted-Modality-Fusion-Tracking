from torch import nn
from .base_promft import BaseProMFT
from ..layers.patch_embed import PatchEmbed
from .base_promft import SCA

class PromptBlock_SCAversion(nn.Module, ):
    def __init__(self, inplanes=None, hide_channel=None, smooth=False):
        super(PromptBlock_SCAversion, self).__init__()
        self.conv0_0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv0_1 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv1x1 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)

        self.sca = SCA()

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """ Forward pass with input x. """
        B, C, W, H = x.shape
        x0 = x[:, 0:int(C/2), :, :].contiguous()
        x0 = self.conv0_0(x0)
        #print("x0:",x0.shape)
        x1 = x[:, int(C/2):, :, :].contiguous()
        x1 = self.conv0_1(x1)
        #print("x1:",x1.shape)

        
        x = self.sca(x0, x1)
        #print("x:",x.shape)

        return self.conv1x1(x)

class ProMFTSCA(BaseProMFT):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer_cls=None,
                 act_layer_cls=None, weight_init='', ce_loc=None, ce_keep_ratio=None, search_size=None, template_size=None,
                 new_patch_size=None,):
        super(ProMFTSCA, self).__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, representation_size=representation_size, distilled=distilled, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, embed_layer=embed_layer, norm_layer_cls=norm_layer_cls, act_layer_cls=act_layer_cls, weight_init=weight_init, ce_loc=ce_loc, ce_keep_ratio=ce_keep_ratio, search_size=search_size, template_size=template_size, new_patch_size=new_patch_size)
        
    def create_prompt_blocks(self):
        blocks = [PromptBlock_SCAversion(inplanes=self.embed_dim, hide_channel=32, smooth=True)
            for i in range(self.depth)]
        return nn.Sequential(*blocks)
    
    def create_prompt_norms(self):
        norms = [self.norm_layer_cls(self.embed_dim) for i in range(self.depth)]
        return nn.Sequential(*norms)