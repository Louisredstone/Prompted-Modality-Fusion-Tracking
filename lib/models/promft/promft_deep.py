import torch
from torch import nn
from .base_promft import BaseProMFT
from ..layers.patch_embed import PatchEmbed
from .base_promft import AFP
from ..utils import combine_tokens, recover_tokens, token2feature, feature2token
from ..layers.attn_blocks import CEBlock, candidate_elimination_prompt

class PromptBlock(nn.Module, ):
    def __init__(self, inplanes=None, hide_channel=None, smooth=False):
        super(PromptBlock, self).__init__()
        self.conv0_0 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv0_1 = nn.Conv2d(in_channels=inplanes, out_channels=hide_channel, kernel_size=1, stride=1, padding=0)
        self.conv1x1 = nn.Conv2d(in_channels=hide_channel, out_channels=inplanes, kernel_size=1, stride=1, padding=0)

        self.afp = AFP()

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        """ Forward pass with input x. """
        B, C, W, H = x.shape # 16, 1536, 8, 8
        x0 = x[:, 0:int(C/2), :, :].contiguous() # Tensor[16, 768, 8, 8]
        x0 = self.conv0_0(x0) # Tensor[16, 32, 8, 8]
        #print("x0:",x0.shape)
        x1 = x[:, int(C/2):, :, :].contiguous()
        x1 = self.conv0_1(x1)
        #print("x1:",x1.shape)

        
        x = self.afp(x0, x1)
        #print("x:",x.shape)

        return self.conv1x1(x)

class ProMFTDeep(BaseProMFT):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer_cls=None,
                 act_layer_cls=None, weight_init='', ce_loc=None, ce_keep_ratio=None, search_size=None, template_size=None,
                 new_patch_size=None,):
        super(ProMFTDeep, self).__init__(img_size=img_size, patch_size=patch_size, in_chans=in_chans, num_classes=num_classes, embed_dim=embed_dim, depth=depth, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, representation_size=representation_size, distilled=distilled, drop_rate=drop_rate, attn_drop_rate=attn_drop_rate, drop_path_rate=drop_path_rate, embed_layer=embed_layer, norm_layer_cls=norm_layer_cls, act_layer_cls=act_layer_cls, weight_init=weight_init, ce_loc=ce_loc, ce_keep_ratio=ce_keep_ratio, search_size=search_size, template_size=template_size, new_patch_size=new_patch_size)
        
    def create_prompt_blocks(self):
        blocks = [PromptBlock(inplanes=self.embed_dim, hide_channel=32, smooth=True) for i in range(self.depth)]
        return nn.Sequential(*blocks)
    
    def create_prompt_norms(self):
        norms = [self.norm_layer_cls(self.embed_dim) for i in range(self.depth)]
        # default: nn.LayerNorm
        return nn.Sequential(*norms)
    
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
    
    def forward_block_i(self, x, z, x_dte, z_dte, i, batch_size, lens_x, lens_z, global_index_t, global_index_s, removed_indexes_s):
        # x: Tensor[16, 320, 768], z: Tensor[16, 64, 768], x_dte: [16, 256, 768], z_dte: [16, 644, 768]
        B = batch_size
        x_ori = x # x_ori: Tensor[16, 320, 768]
        # recover x to go through prompt blocks
        lens_z_new = global_index_t.shape[1] # can be replaced with lens_z.
        lens_x_new = global_index_s.shape[1] # can be replaced with lens_x.
        z = x[:, :lens_z_new] # SHITCODE: z UNUSED
        x = x[:, lens_z_new:] # now z: Tensor[16, 64, 768], x: Tensor[16, 256, 768]
        if removed_indexes_s and removed_indexes_s[0] is not None:
            removed_indexes_cat = torch.cat(removed_indexes_s, dim=1)
            pruned_lens_x = lens_x - lens_x_new
            pad_x = torch.zeros([B, pruned_lens_x, x.shape[2]], device=x.device)
            x = torch.cat([x, pad_x], dim=1)
            index_all = torch.cat([global_index_s, removed_indexes_cat], dim=1)
            C = x.shape[-1]
            x = torch.zeros_like(x).scatter_(dim=1, index=index_all.unsqueeze(-1).expand(B, -1, C).to(torch.int64), src=x)
        x = recover_tokens(x, lens_z_new, lens_x, mode=self.cat_mode) # x: Tensor[16, 256, 768]
        x = torch.cat([z, x], dim=1) # x: Tensor[16, 320, 768] with z on its head

        # prompt
        x = self.prompt_norms[i - 1](x) # x: Tensor[16, 320, 768]
        z_tokens = x[:, :lens_z, :] # Tensor[16, 64, 768]
        x_tokens = x[:, lens_z:, :] # Tensor[16, 256, 768]
        z_feat = token2feature(z_tokens) # Tensor[16, 768, 8, 8]
        x_feat = token2feature(x_tokens) # Tensor[16, 768, 16, 16]

        z_prompted, x_prompted = z_dte, x_dte # Tensor[16, 64, 768], Tensor[16, 256, 768]
        z_prompted = self.prompt_norms[i](z_prompted) # Tensor[16, 64, 768]
        x_prompted = self.prompt_norms[i](x_prompted) # Tensor[16, 256, 768]
        z_prompt_feat = token2feature(z_prompted) # Tensor[16, 768, 8, 8]
        x_prompt_feat = token2feature(x_prompted) # Tensor[16, 768, 16, 16]

        # z_feat = torch.cat([z_feat, z_prompt_feat], dim=1)
        # x_feat = torch.cat([x_feat, x_prompt_feat], dim=1)
        # #print("prompt block input:",x_feat.shape,z_feat.shape)
        # z_feat = self.prompt_blocks[i](z_feat)
        # x_feat = self.prompt_blocks[i](x_feat)

        # z = feature2token(z_feat)
        # x = feature2token(x_feat)
        # z_prompted, x_prompted = z, x

        # x = combine_tokens(z, x, mode=self.cat_mode)
        # # re-conduct CE
        # x = x_ori + candidate_elimination_prompt(x, global_index_t.shape[1], global_index_s)
        
        # return z, x
        
        # z_feat = torch.cat([z_feat, z_prompt_feat], dim=1)
        # x_feat = torch.cat([x_feat, x_prompt_feat], dim=1)
        # #print("prompt block input:",x_feat.shape,z_feat.shape)
        # z_feat = self.prompt_blocks[i](z_feat)
        # x_feat = self.prompt_blocks[i](x_feat)

        # z = feature2token(z_feat)
        # x = feature2token(x_feat)
        # z_prompted, x_prompted = z, x
        
        z_prompted = feature2token(
            self.prompt_blocks[i](
                torch.cat([z_feat, z_prompt_feat], dim=1)
                )
            ) # Tensor[16, 64, 768]
        x_prompted = feature2token(
            self.prompt_blocks[i](
                torch.cat([x_feat, x_prompt_feat], dim=1)
                )
            ) # Tensor[16, 256, 768]

        # re-conduct CE
        x_final = x_ori + candidate_elimination_prompt(
            tokens=combine_tokens(
                template_tokens=z_prompted, 
                search_tokens=x_prompted, 
                mode=self.cat_mode), # Tensor[16, 320, 768]
            lens_t=global_index_t.shape[1], 
            global_index=global_index_s) # output of candidate_elimination_prompt: Tensor[16, 320, 768]
        # H_{l}' = H_l + P_{l+1}
        
        return z_prompted, x_final