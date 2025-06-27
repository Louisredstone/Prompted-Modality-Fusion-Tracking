from matplotlib import pyplot as plt
import cv2

###### pareto plot

print("Drawing pareto plot...")

pareto = {
    'ProMFT (ours)': {
        'total_params': 94072581,
        'score': {
            'LasHeR': {'PR': 65.2, 'SR': 57.6},
            'DepthTrack': {'F1': 62.2, 'Pr': 62.5, 'Re': 64.7},
            'VisEvent': {'PR': 84.2, 'SR': 61.7}
        }
    },
    'ViPT': {
        'total_params': 93.5e6,
        'score': {
            'LasHeR': {'PR': 65.1, 'SR': 52.5},
            'DepthTrack': {'F1': 59.4, 'Pr': 59.1, 'Re': 59.6},
            'VisEvent': {'PR': 75.8, 'SR': 59.2}
        }
    },
    'BAT': {
        'total_params': None,
        'score': {
            'LasHeR': {'PR': 70.2, 'SR': 56.3},
            # 'DepthTrack': {'F1': 0, 'Pr': 0, 'Re': 0},
            # 'VisEvent': {'PR': 0, 'SR': 0}
        }
    },
    'TATrack': {
        'total_params': None,
        'score': {
            'LasHeR': {'PR': 70.2, 'SR': 56.1},
            # 'DepthTrack': {'F1': 0, 'Pr': 0, 'Re': 0},
            # 'VisEvent': {'PR': 0, 'SR': 0}
        }
    },
    'CFBT': {
        'total_params': None,
        'score': {
            'LasHeR': {'PR': 73.2, 'SR': 58.4},
            # 'DepthTrack': {'F1': 0, 'Pr': 0, 'Re': 0},
            # 'VisEvent': {'PR': 0, 'SR': 0}
        }
    }
}

plt.figure(figsize=(10, 5))

for method, info in pareto.items():
    if info['total_params'] is not None:
        plt.plot(info['total_params'], info['score']['LasHeR']['PR'], 'o', label=method)

plt.xlabel('Total Parameters')
plt.ylabel('LasHeR PR')
plt.legend()
plt.savefig('figures/LasHeR_PR.png')

plt.close()

###### figure for the diagram

print("Drawing figure for the diagram...")

from lib.data.data_resource import LasHeR

data_resource = LasHeR('datasets/LasHeR', 'train')

# seq_name, chosen_id = 'cycleman', 35
# seq_name, chosen_id = 'man_with_black_clothes2', 83
# seq_name, chosen_id = 'orangegirl', 104
# seq_name, chosen_id = 'redbaginbike', 41
# seq_name, chosen_id = 'schoolbus', 63
seq_name, chosen_id = 'whiteboyback', 37

seq = data_resource.getSeqByName(seq_name)
frame = seq.load_frame(chosen_id)
rgb_image_chw, aux_image_chw = frame[:3], frame[3:]
rgb_image_hwc = rgb_image_chw.transpose(1, 2, 0).copy()
aux_image_hwc = aux_image_chw.transpose(1, 2, 0).copy()
cv2.imwrite('figures/'+seq_name+'_rgb.png', rgb_image_hwc)
cv2.imwrite('figures/'+seq_name+'_aux.png', aux_image_hwc)
gt_bbox = seq.bboxes_ltrb[chosen_id]
gt_bbox = [int(p) for p in gt_bbox]

cv2.rectangle(rgb_image_hwc, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (0, 0, 255), 2)
cv2.rectangle(aux_image_hwc, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (0, 0, 255), 2)
cv2.imwrite('figures/'+seq_name+'_rgb_boxed.png', rgb_image_hwc)
cv2.imwrite('figures/'+seq_name+'_aux_boxed.png', aux_image_hwc)

###### qualitative results for the paper
print("Drawing qualitative results for the paper...")