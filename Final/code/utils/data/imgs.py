import abc
import glob
import hashlib
import logging
import os
import subprocess
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage import io
from PIL import Image
from skorch.utils import to_numpy
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from tqdm import tqdm

from utils.helpers import set_seed

from .helpers import DIR_DATA, preprocess, random_translation, train_dev_split

DIR = os.path.abspath(os.path.dirname(__file__))
COLOUR_BLACK = torch.tensor([0.0, 0.0, 0.0])
COLOUR_WHITE = torch.tensor([1.0, 1.0, 1.0])
COLOUR_BLUE = torch.tensor([0.0, 0.0, 1.0])
DATASETS_DICT = {
    "mnist": "MNIST",
    "fashionmnist": "FashionMNIST",
    "dem30m": "dem30m",
    "eurosat": "EuroSAT"
}
DATASETS = list(DATASETS_DICT.keys())

logger = logging.getLogger(__name__)


# HELPERS
def get_train_test_img_dataset(dataset):
    """Return the correct instantiated train and test datasets."""
    try:
        train_dataset = get_dataset(dataset)(split="train")
        test_dataset = get_dataset(dataset)(split="test")
    except TypeError as e:
        train_dataset, test_dataset = train_dev_split(
            get_dataset(dataset)(), dev_size=0.1, is_stratify=False
        )

    return train_dataset, test_dataset


def get_dataset(dataset):
    """Return the correct uninstantiated datasets."""
    dataset = dataset.lower()
    try:
        # eval because stores name as string in order to put it at top of file
        return eval(DATASETS_DICT[dataset])
    except KeyError:
        raise ValueError("Unkown dataset: {}".format(dataset))


def get_img_size(dataset):
    """Return the correct image size."""
    return get_dataset(dataset).shape


def get_test_upscale_factor(dataset):
    """Return the correct image size."""
    try:
        dataset = get_dataset(dataset)
        return dataset.shape_test[-1] / dataset.shape[-1]
    except (AttributeError, ValueError):
        return 1


# TORCHVISION DATASETS
class MNIST(datasets.MNIST):
    """MNIST wrapper. Docs: `datasets.MNIST.`

    Parameters
    ----------
    root : str, optional
        Path to the dataset root. If `None` uses the default one.

    split : {'train', 'test', "extra"}, optional
        According dataset is selected.

    kwargs:
        Additional arguments to `datasets.MNIST`.
    """

    shape = (1, 32, 32)
    n_classes = 10
    missing_px_color = COLOUR_BLUE
    name = "MNIST"

    def __init__(
        self, root=DIR_DATA, split="train", logger=logging.getLogger(__name__), **kwargs
    ):

        if split == "train":
            transforms_list = [transforms.Resize(32), transforms.ToTensor()]
        elif split == "test":
            transforms_list = [transforms.Resize(32), transforms.ToTensor()]
        else:
            raise ValueError("Unkown `split = {}`".format(split))

        super().__init__(
            root,
            train=split == "train",
            download=True,
            transform=transforms.Compose(transforms_list),
            **kwargs
        )

        self.targets = to_numpy(self.targets)

class FashionMNIST(datasets.FashionMNIST):
    """MNIST wrapper. Docs: `datasets.MNIST.`

    Parameters
    ----------
    root : str, optional
        Path to the dataset root. If `None` uses the default one.

    split : {'train', 'test', "extra"}, optional
        According dataset is selected.

    kwargs:
        Additional arguments to `datasets.FashionMNIST`.
    """

    shape = (1, 32, 32)
    n_classes = 10
    missing_px_color = COLOUR_BLUE
    name = "FashionMNIST"

    def __init__(
        self, root=DIR_DATA, split="train", logger=logging.getLogger(__name__), **kwargs
    ):

        if split == "train":
            transforms_list = [transforms.Resize(32), transforms.ToTensor()]
        elif split == "test":
            transforms_list = [transforms.Resize(32), transforms.ToTensor()]
        else:
            raise ValueError("Unkown `split = {}`".format(split))

        super().__init__(
            root,
            train=split == "train",
            download=True,
            transform=transforms.Compose(transforms_list),
            **kwargs
        )

        self.targets = to_numpy(self.targets)

# EXTERNAL DATASETS (not torchvision)
class ExternalDataset(Dataset, abc.ABC):
    """Base Class for external datasets.

    Parameters
    ----------
    root : string
        Root directory of dataset.

    transforms_list : list
        List of `torch.vision.transforms` to apply to the data when loading it.
    """

    def __init__(self, root, transforms_list=[], logger=logging.getLogger(__name__)):
        self.dir = os.path.join(root, self.name)
        self.train_data = os.path.join(self.dir, type(self).files["train"])
        self.transforms = transforms.Compose(transforms_list)
        self.logger = logger

        if not os.path.isdir(self.dir):
            self.logger.info("Downloading {} ...".format(str(type(self))))
            self.download()
            self.logger.info("Finished Downloading.")

    def __len__(self):
        return len(self.imgs)

    @abc.abstractmethod
    def __getitem__(self, idx):
        """Get the image of `idx`.

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `shape`.
        """
        pass

    @abc.abstractmethod
    def download(self):
        """Download the dataset. """
        pass

class SingleImage(Dataset):
    def __init__(
        self,
        img,
        resize=None,
        transforms_list=[transforms.ToTensor()],
        missing_px_color=COLOUR_BLACK,
    ):

        self.missing_px_color = missing_px_color
        self.img = transforms.ToPILImage()(img)
        if resize is not None:
            self.img = transforms.Resize(resize)(self.img)

        self.shape = transforms.ToTensor()(self.img).shape
        self.transforms = transforms.Compose(transforms_list)

    def __getitem__(self, i):
        return self.transforms(self.img).float(), 0

    def __len__(self):
        return 1

class dem30m(ExternalDataset):
    """
    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] 

    """

    files = {
        "train": "DEM_Train",
        "test": "DEM_Test"}
    shape = (1, 64, 64)
    missing_px_color = COLOUR_BLACK
    n_classes = 0  # not classification
    name = "dem30m"

    def __init__(self, root=DIR_DATA, **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)

        self.imgs = glob.glob(self.train_data + "/*")

    def download(self):
        self.preprocess()

    def preprocess(self):
        self.logger.info("Resizing dem30m ...")
        preprocess(self.train_data, size=type(self).shape[1:])

    def __getitem__(self, idx):
        """Get the image of `idx`

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `shape`.

        placeholder :
            Placeholder value as their are no targets.
        """
        img_path = self.imgs[idx]
        # img values already between 0 and 255
        # img = plt.imread(img_path)
        img = io.imread(img_path)
        # print(img.shape)
        # put each pixel in [0.,1.] and reshape to (C x H x W)
        img = self.transforms(img)
        img_min = torch.min(img)
        img_max = torch.max(img)
        img = (img - img_min) / (img_max - img_min + 1)
        # img = img * 255
        # print(img.shape)
        # no label so return 0 (note that can't return None because)
        # dataloaders requires so
        return img, 0

class EuroSAT(ExternalDataset):
    """
    Parameters
    ----------
    root : string
        Root directory of dataset.

    References
    ----------
    [1] 

    """

    files = {
        "train": "Industrial",}
    shape = (3, 64, 64)
    missing_px_color = COLOUR_BLACK
    n_classes = 0  # not classification
    name = "EuroSAT"

    def __init__(self, root=DIR_DATA, **kwargs):
        super().__init__(root, [transforms.ToTensor()], **kwargs)

        self.imgs = glob.glob(self.train_data + "/*")

    def download(self):
        self.preprocess()

    def preprocess(self):
        self.logger.info("Resizing EuroSAT ...")
        preprocess(self.train_data, size=type(self).shape[1:])

    def __getitem__(self, idx):
        """Get the image of `idx`

        Return
        ------
        sample : torch.Tensor
            Tensor in [0.,1.] of shape `shape`.

        placeholder :
            Placeholder value as their are no targets.
        """
        img_path = self.imgs[idx]
        # img values already between 0 and 255
        img = plt.imread(img_path)

        # put each pixel in [0.,1.] and reshape to (C x H x W)
        img = self.transforms(img)

        # no label so return 0 (note that can't return None because)
        # dataloaders requires so
        return img, 0
