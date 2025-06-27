import os
import sys
import numpy as np

data_dir = '/home/public/dataset/RGBT_Datasets/LasHeR/TrainingSet/trainingset/'
name_list = os.listdir(data_dir)
# print(name_list[:10])
print('####################################################')
for name in name_list:
    name_dir = os.path.join(data_dir, name)
    os.chdir(name_dir)
    os.chdir('visible')
    visible_dir = os.path.join(name_dir, 'visible')
    print(visible_dir)
    img_list = os.listdir(visible_dir)
    n = len(img_list)
    print(n)
    sorted_img_list = np.sort(img_list)
    # print(np.sort(img_list))
    for i in range(n):
        os.system('mv '+sorted_img_list[i]+' {:06d}.jpg'.format(i))
    infra_dir = os.path.join(name_dir, 'infrared')
    os.chdir(infra_dir)
    img_list = os.listdir(infra_dir)
    n = len(img_list)
    print(n)
    sorted_img_list = np.sort(img_list)
    for i in range(n):
        os.system('mv '+sorted_img_list[i]+' {:06d}.jpg'.format(i))
    os.chdir('../..')
    # os.system('pwd')