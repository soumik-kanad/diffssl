import os
import torchvision.datasets as datasets
from PIL import Image


class ImageNet(datasets.ImageFolder):
    def __init__(self, root, split='train', transform=None):
        super(ImageNet, self).__init__(root=os.path.join(root, split),
                                         transform=None)
        self.transform = transform 
        self.split = split
    
    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        path, target = self.imgs[index]
        with open(path, 'rb') as f:
            img = Image.open(f).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        
        return img, {"y": target}