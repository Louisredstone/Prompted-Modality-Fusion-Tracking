import numpy as np

def calc_rect_int(A, B):
    # Calculate overlap of two rectangles
    leftA = A[:, 0]
    bottomA = A[:, 1]
    rightA = leftA + A[:, 2] - 1
    topA = bottomA + A[:, 3] - 1

    leftB = B[:, 0]
    bottomB = B[:, 1]
    rightB = leftB + B[:, 2] - 1
    topB = bottomB + B[:, 3] - 1

    tmp = (np.maximum(0, np.minimum(rightA, rightB) - np.maximum(leftA, leftB) + 1)) * \
          (np.maximum(0, np.minimum(topA, topB) - np.maximum(bottomA, bottomB) + 1))
    areaA = A[:, 2] * A[:, 3]
    areaB = B[:, 2] * B[:, 3]
    overlap = tmp / (areaA + areaB - tmp)
    return overlap