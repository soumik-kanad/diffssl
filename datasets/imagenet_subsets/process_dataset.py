import os
import shutil

imagenet_root = ''
imagenet_10percent_root = ''

with open('10percent.txt', 'r') as f:
    lines = f.readlines()


for line in lines:
    cls = line.split('_')[0]
    #print(os.path.join(imagenet_root, cls, line))
    src = os.path.join(imagenet_root, cls, line.strip())
    dst = os.path.join(imagenet_10percent_root, cls, line.strip())
    #print(path, os.path.isfile(path))
    assert os.path.isfile(src)
    if cls not in os.listdir(imagenet_10percent_root):
        os.makedirs(os.path.join(imagenet_10percent_root, cls))
    shutil.copy(src, dst)
