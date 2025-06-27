"""
Basic ViPT model.
"""
import math
import os
from typing import List
from timm.models.layers import to_2tuple
import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from .layers.head import build_box_head
# from .vit.vit_prompt import VisionTransformerWithPrompt
# from ..vit.vit_ce_prompt import vit_base_patch16_224_ce_prompt
# from .vit.vit_ce_prompt import VisionTransformerCEWithPrompt
from ..utils.box_ops import box_xyxy_to_cxcywh
from .promft import ProMFTDeep, ProMFTShallow, ProMFTNaive, ProMFTDifferential, ProMFTNaiveShallow
from .no_prompt_wrapper import NoAUXWrapper
from ..config import MainConfig

class Tracker(nn.Module):
    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer # TODO: add prompt. Current implementation (mis)makes prompt frameworks as backbones. Ideally, we should have a separate prompt module.
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]
        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError

    @staticmethod
    def build(CONFIG: MainConfig, training=True):
        current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
        pretrained_path = os.path.join(current_dir, '../../../pretrained_models')  # use pretrained OSTrack as initialization
        if CONFIG.MODEL.PRETRAIN_FILE and ('OSTrack' not in CONFIG.MODEL.PRETRAIN_FILE) and training:
            pretrained = os.path.join(pretrained_path, CONFIG.MODEL.PRETRAIN_FILE)
        else:
            pretrained = ''

        if CONFIG.TRAIN.PROMPT.TYPE in ["promft_deep", "promft"]: cls = ProMFTDeep
        elif CONFIG.TRAIN.PROMPT.TYPE in ["promft_shallow"]: cls = ProMFTShallow
        elif CONFIG.TRAIN.PROMPT.TYPE in ["promft_naive"]: cls = ProMFTNaive
        elif CONFIG.TRAIN.PROMPT.TYPE in ["promft_differential"]: cls = ProMFTDifferential
        elif CONFIG.TRAIN.PROMPT.TYPE in ["promft_naive_shallow"]: cls = ProMFTNaiveShallow
        elif CONFIG.TRAIN.PROMPT.TYPE in ["no_aux_wrapper"]: cls = NoAUXWrapper
        elif CONFIG.TRAIN.PROMPT.TYPE in ["vipt"]: raise NotImplementedError()
        else: raise ValueError(f"illegal prompt type {CONFIG.TRAIN.PROMPT.TYPE}")

        # SHITCODE: 'backbone' is a misnomer, it should be 'transformer' or 'framework' or 'transformer_with_prompt'.
        if CONFIG.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_prompt':
            backbone = cls.build_base(pretrained, drop_path_rate=CONFIG.TRAIN.DROP_PATH_RATE,
                                                search_size=to_2tuple(CONFIG.DATA.SEARCH.SIZE),
                                                template_size=to_2tuple(CONFIG.DATA.TEMPLATE.SIZE),
                                                new_patch_size=CONFIG.MODEL.BACKBONE.STRIDE
                                                )
            hidden_dim = backbone.embed_dim
            patch_start_index = 1

        elif CONFIG.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce_prompt':
            backbone = cls.build_base(pretrained, drop_path_rate=CONFIG.TRAIN.DROP_PATH_RATE,
                                            ce_loc=CONFIG.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=CONFIG.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            search_size=to_2tuple(CONFIG.DATA.SEARCH.SIZE), # (256, 256)
                                            template_size=to_2tuple(CONFIG.DATA.TEMPLATE.SIZE),
                                            new_patch_size=CONFIG.MODEL.BACKBONE.STRIDE
                                            )
            hidden_dim = backbone.embed_dim
            patch_start_index = 1

        else:
            raise NotImplementedError()
        """For prompt no need, because we have OSTrack as initialization"""
        # backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

        box_head = build_box_head(CONFIG, hidden_dim)

        model = Tracker(
            backbone,
            box_head,
            aux_loss=False,
            head_type=CONFIG.MODEL.HEAD.TYPE,
        )

        if 'OSTrack' in CONFIG.MODEL.PRETRAIN_FILE and training:
            checkpoint = torch.load(CONFIG.MODEL.PRETRAIN_FILE, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            print('Load pretrained model from: ' + CONFIG.MODEL.PRETRAIN_FILE)
            #print(f"missing_keys: {missing_keys}")
            #print(f"unexpected_keys: {unexpected_keys}")

        return model
