import os
import gzip
from copy import deepcopy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset

class TransformTensorDataset(Dataset):
    """TensorDataset with support of transforms.
    https://stackoverflow.com/questions/55588201/pytorch-transforms-on-tensordataset
    """
    def __init__(self, *tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.tensors[0].size(0)

class UndersampledUltrasoundDataset3D(Dataset):
    def __init__(self, path, transform=None, mode="mri", undersample_width=(2, 0, 0), maxsamples=None):

        self.path = path
        self.maxsamples=maxsamples
        self.transform = transform

        if len(undersample_width) == 3:
            self.undersample_width = undersample_width
        else:
            raise ValueError("Undersample_width must be a tuple of length 3. To keep sampling along a dimension set tuple item to 1.")

        if mode.lower() in ["mri", "vp"]:
            self.mode=mode
            self.fill = 0. if mode.lower() == "mri" else 1500.
        else:
            raise ValueError("mode must be 'mri' or 'vp', but found '{}'".format(mode))

        # Get image paths
        self.data_paths = []
        self._get_image_paths(path, maxsamples)

        return None

    def _get_image_paths(self, path, maxsamples=None):
        prefix = "m" if self.mode == "mri" else "vp"
        suffix = ".npy.gz"
        
        # Loop through folders and subfolders
        for subdir, _, files in os.walk(path):
            for filename in files:
                if filename.lower().startswith(prefix) and filename.lower().endswith(suffix):
                    self.data_paths.append(os.path.join(subdir, filename))
        self.data_paths = self.data_paths[:maxsamples]
        return None

    def __getitem__(self, index):
        # Item path
        y_path = self.data_paths[index]

        # Unzip and read data
        with gzip.open(y_path, 'rb') as f:
            y = torch.from_numpy(np.load(f))

        # Apply transform
        if self.transform:
            y = self.transform(y)

        # Undersample data
        x = torch.zeros_like(y) + self.fill
        dx, dy, dz = self.undersample_width
        idx = [i*dx for i in range(int(x.shape[0]/dx) + 1)]
        if idx[-1] == x.shape[0]:
            idx = idx[:-1]
        idy = [i*dy for i in range(int(x.shape[1]/dy) + 1)]
        if idy[-1] == x.shape[1]:
            idy = idy[:-1]
        idz = [i*dz for i in range(int(x.shape[2]/dz) + 1)]
        if idz[-1] == x.shape[2]:
            idz = idz[:-1]
        idmesh = torch.meshgrid(torch.tensor(idx), torch.tensor(idy), torch.tensor(idz))
        x[idmesh] = y[idmesh]

        return x, y

    def __len__(self):
        return len(self.data_paths)

    def __str__(self):
        dic = {}
        dic["name"] = self.__class__.__name__
        dic.update(self.__dict__)
        dic.pop("data_paths")
        dic["len"] = self.__len__()
        return "{}".format(dic)

    def info(self, nsamples=30):
        sample = self.__getitem__(0)[0]
        idx = torch.randint(0, self.__len__(), [nsamples])
        arr = torch.empty_like(sample).unsqueeze(0).repeat(nsamples, 1, 1, 1)
        for i in range(nsamples):
            arr[i] = self.__getitem__(idx[i])[0]
        stats = {"max": arr.max(), "min": arr.min(), "mean": arr.mean(), "std": arr.std(), "shape":sample.shape}

        return stats


class MaskedUltrasoundDataset2D(Dataset):
    def __init__(self, path, transform=None, mode="mri", mask=None, maxsamples=None):
        super(MaskedUltrasoundDataset2D, self).__init__()
        self.path = path
        self.maxsamples=maxsamples
        self.transform = transform
        self.mask = mask

        if mode.lower() in ["mri", "vp"]:
            self.mode=mode
        else:
            raise ValueError("mode must be 'mri' or 'vp', but found '{}'".format(mode))

        # Get image paths
        self.data_paths = []
        self._get_image_paths(path, maxsamples)

        return None

    def _get_image_paths(self, path, maxsamples=None):
        prefix = "m" if self.mode == "mri" else "vp"
        suffix = ".npy"
        
        # Loop through folders and subfolders
        for subdir, _, files in os.walk(path):
            for filename in files:
                if filename.lower().startswith(prefix) and filename.lower().endswith(suffix):
                    self.data_paths.append(os.path.join(subdir, filename))
        self.data_paths = self.data_paths[:maxsamples]
        return None

    def __getitem__(self, index):
        # Item path
        y_path = self.data_paths[index]

        # Read data
        y = torch.from_numpy(np.load(y_path))

        # Apply transform
        if self.transform:
            y = self.transform(y)

        if self.mask is not None:
            x = self.mask * y
            return x, y

        else:
            return y


    def __len__(self):
        return len(self.data_paths)

    def __str__(self):
        dic = {}
        dic["name"] = self.__class__.__name__
        dic.update(self.__dict__)
        dic.pop("data_paths")
        dic["len"] = self.__len__()
        return "{}".format(dic)

    def info(self, nsamples=None):
        nsamples = self.__len__() if nsamples is None else nsamples
        sample = self.__getitem__(0)[0]
        idx = torch.randint(0, self.__len__(), [nsamples])
        arr = torch.empty_like(sample).unsqueeze(0).repeat(nsamples, 1, 1, 1)
        for i in range(nsamples):
            arr[i] = self.__getitem__(idx[i])[0]
        stats = {"max": arr.max().item(), "min": arr.min().item(), "mean": arr.mean().item(), "std": arr.std().item(), "shape":sample.shape}

        return stats

