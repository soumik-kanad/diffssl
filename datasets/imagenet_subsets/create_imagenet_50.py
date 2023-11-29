import shutil
import os

folder_name_file = '/path/to/imagenet_50.txt'
imagenet_root = '/path/to/imagenet'
inet50_dest = '/path/to/imagenet_50'
splits = ['train', 'val']

with open(folder_name_file, 'r') as read_file:
    lines = read_file.readlines()

class_folders = []
for line in lines:
    class_folders.append(line.strip().split(' ')[0].strip())

for split in splits:
    if split not in os.listdir(os.path.join(inet50_dest)):
        os.makedirs(os.path.join(inet50_dest, split))
    for class_folder in class_folders:
        if class_folder not in os.listdir(os.path.join(inet50_dest, split)):
            os.makedirs(os.path.join(inet50_dest, split, class_folder))
        for filename in os.listdir(os.path.join(imagenet_root, split, class_folder)):
            shutil.copy(os.path.join(imagenet_root, split, class_folder, filename), os.path.join(inet50_dest, split, class_folder))