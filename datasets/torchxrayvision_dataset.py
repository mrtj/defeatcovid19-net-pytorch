import numpy as np
from torch.utils.data import Dataset

class TorchXRayVisionDataset(Dataset):
    '''
    TorchXRayVision Dataset

    The TorchXRayVisionDataset class in wraps a torchxrayvision.dataset
    instance to a dataset used by this project, so all public datasets
    supported by the `torchxrayvision`_ library can be used to train the model.

    To use this dataset wrapper:
     - install torchxrayvision with `pip install torchxrayvision`,
     - download one of the datasets supported by torchxrayvision
       (see `torchxrayvision.datasets`_ module)
     - create an instance of one of the `torchxrayvision.datasets`_ classes,
     - pass this instance to  the TorchXRayVisionDataset_ initializer.

    More info: https://github.com/mlmed/torchxrayvision

    Example usage of TorchXRayVisionDataset::

        import torchvision
        import torchxrayvision as xrv

        transform = torchvision.transforms.Compose([xrv.datasets.XRayCenterCrop(),
                                                    xrv.datasets.XRayResizer(size)])

        kaggle_dataset = xrv.datasets.Kaggle_Dataset(
            imgpath='dataset_folder/image_folder',
            csvpath='dataset_folder/stage_2_train_labels.csv',
            transform=transform)

        dataset = TorchXRayVisionDataset(kaggle_dataset, balance=True)

    .. _torchxrayvision: https://github.com/mlmed/torchxrayvision
    .. _torchxrayvision.datasets: https://github.com/mlmed/torchxrayvision/blob/master/torchxrayvision/datasets.py


    Parameters
    ----------
    xrv_dataset : torchxrayvision.datasets.Dataset
        The torchxrayvision dataset instance.

    pathologies : list<str>
        The list of pathology labels that should be evaluated as True label.
        A subset of torchxrayvision.datasets.Dataset.pathologies property of 
        your torchxrayvision dataset instance.

    augment : callable
        If not None, will be called with the image instance.

    balance : bool
        If set to True, create a balanced dataset by undersampling the category
        class with higher number of instances.
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
