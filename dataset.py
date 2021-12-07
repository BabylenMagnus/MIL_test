import os
import cv2

from torch.utils.data import Dataset


class Cifar100(Dataset):
    def __init__(self, root='CIFAR100/TRAIN'):
        self.root = root
        self.list_of_data = []

        for root, _, files in os.walk(self.root):
            for file in files:
                self.list_of_data.append(os.path.join(root.split('/')[-1], file))

        self.classes = os.listdir(self.root)

    def __len__(self):
        return len(self.list_of_data)

    def __getitem__(self, item):
        path = self.list_of_data[item]
        label = path.split('/')[0]
        label = self.classes.index(label)

        img = cv2.imread(os.path.join(self.root, path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img = img.transpose(-1, 0, 1)  # from numpy to torch image

        return img, label
