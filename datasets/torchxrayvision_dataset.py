import cv2
import pandas as pd
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

import torchxrayvision as xrv

class TorchXRayVisionDataset(Dataset):
    ''' This class is an adapter for torchxrayvision datasets.
    
    More info: https://github.com/mlmed/torchxrayvision
    '''

    def __init__(self, xrv_dataset, pathologies=['Pneumonia'], augment=None, balance=False):
        super(TorchXRayVisionDataset, self).__init__()
        print('{} initialized with xrv_dataset={}'.format(self.__class__.__name__, xrv_dataset.__class__.__name__))
        self.xrv_dataset = xrv_dataset
        self.augment = augment

        # determinate binary labels
        pathology_mask = np.zeros(len(xrv_dataset.pathologies))
        for pathology in pathologies:
            idx = xrv_dataset.pathologies.index(pathology)
            pathology_mask[idx] = 1
        labels_mask = np.any(np.logical_and(xrv_dataset.labels, pathology_mask), axis=1)

        if balance:
            pos_indices = np.arange(len(labels_mask))[labels_mask]
            neg_indices = np.arange(len(labels_mask))[~labels_mask]
            if len(pos_indices) < len(neg_indices):
                neg_indices = np.random.choice(neg_indices, len(pos_indices))
            else:
                pos_indices = np.random.choice(pos_indices, len(neg_indices))

            self.labels = np.concatenate((
                np.zeros(len(neg_indices)),
                np.ones(len(pos_indices))
            )).reshape(-1, 1)
            self.item_indices = np.concatenate((neg_indices, pos_indices))
        else:
            self.labels = labels_mask.astype(int)
            self.item_indices = np.arange(len(self.labels))

        print("Dataset size: {}".format(len(self.labels)))
        print("  Number of positive cases: {}".format(sum(self.labels)))
        print("  Number of negative cases: {}".format(len(self.labels) - sum(self.labels)))
    
    def __getitem__(self, index):

        item = self.xrv_dataset[self.item_indices[index]]
        label = self.labels[index]
        img = item['PA']
            
        # xrv works with [-1024; 1024] value range. scale it to [0; 255]
        xrv_min = -1024
        xrv_max = 1024
        img = ((img - xrv_min) / (xrv_max - xrv_min) * 255).astype('uint8')

        if img.shape[0] == 1:
            img = img.repeat(3, axis=0)
        if self.augment is not None:
            img = self.augment(img)

        return img, label.item()

    def __len__(self):
        return self.item_indices.shape[0]
