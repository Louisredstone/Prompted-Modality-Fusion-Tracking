from ...models.tracker import Tracker
from ..tracker.basetracker import BaseTracker
import torch
from ..utils.hann import hann2d
from ...data.utils.processing_utils import sample_target
# for debug
import cv2
import numpy as np
from numpy import ndarray, uint8
#import vot
from ..tracker.data_utils import PreprocessorMM
from ...utils.box_ops import clip_box
from ...utils.ce_utils import generate_mask_cond


class ViPT_Test_Wrapper(BaseTracker): # SHITCODE
    def __init__(self, CONFIG):
        super(ViPT_Test_Wrapper, self).__init__(CONFIG)
        # network = build_ostrack_with_prompt(params.cfg, training=False)
        network = Tracker.build(CONFIG, training=False)
        ckpt = torch.load(CONFIG.TEST.CHECKPOINT, map_location='cpu')
        network.load_state_dict(ckpt['net'], strict=True)
        # network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.CONFIG = CONFIG
        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = PreprocessorMM()
        self.state = None

        self.feat_sz = self.CONFIG.TEST.SEARCH_SIZE // self.CONFIG.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()
        
        self.use_visdom = False #params.debug
        self.debug = self.CONFIG.GENERAL.DEBUG
        self.frame_id = 0

    def initialize(self, image: ndarray[(6, 'H', 'W'), uint8], region):
        _, self.H, self.W = image.shape
        image_hwc = image.transpose((1, 2, 0))
        gt_bbox_np = np.array(region).astype(np.float32)
        info = {'init_bbox': list(gt_bbox_np)}  # input must be (x,y,w,h)
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr  = sample_target(image, info['init_bbox'], self.template_factor,
                                                    output_sz=self.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr)
        with torch.no_grad():
            self.z_tensor = template

        self.box_mask_z = None
        if self.CONFIG.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.CONFIG, 1, template.device, template_bbox)

        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.CONFIG.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image: ndarray[(6, 'H', 'W'), uint8], info: dict = None):
        _, H, W = image.shape
        image_hwc = image.transpose((1, 2, 0))
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.search_factor,
                                                                output_sz=self.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr)

        with torch.no_grad():
            x_tensor = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_tensor, search=x_tensor, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        response = self.output_window * pred_score_map
        pred_boxes, best_score = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'], return_score=True)
        max_score = best_score[0][0].item()
        pred_boxes = pred_boxes.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)

        # for debug
        if self.debug:
            x1, y1, w, h = self.state
            image_hwc_bgr = cv2.cvtColor(image_hwc[:,:,:3], cv2.COLOR_RGB2BGR)
            cv2.rectangle(image_hwc_bgr, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 0, 255), thickness=2)
            cv2.putText(image_hwc_bgr, 'max_score:' + str(round(max_score, 3)), (40, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 255, 255), 2)
            cv2.imshow('debug_vis', image_hwc_bgr)
            cv2.waitKey(1)

        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            output = {"target_bbox": self.state,
                    "all_boxes": all_boxes_save,
                    "best_score": max_score}
        else:
            output = {"target_bbox": self.state,
                    "best_score": max_score}
        pred_bbox, pred_score = output['target_bbox'], output['best_score']
        return pred_bbox, pred_score

    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)


def get_wrapper_class():
    return ViPT_Test_Wrapper
