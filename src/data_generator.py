import os
from glob import glob
import json
from typing import Tuple, List

import cv2
import torch
import numpy as np
import albumentations as A
from torch.utils.data import Dataset

from config import INPUT_SHAPE_IMAGE, JSON_NAME, AUGMENTATION_DATA, PATH_DATA, MEAN_STANDART_DEVIATION


class CustomDataset(Dataset):
    def __init__(self, data_path: str = PATH_DATA, json_name: str = JSON_NAME, shuffle_data: bool = False,
                 augmentation_data: bool = AUGMENTATION_DATA, image_shape: Tuple[int, int, int] = INPUT_SHAPE_IMAGE,
                 mean_std: Tuple[List, List]= MEAN_STANDART_DEVIATION, is_train: str = 'val'):
        """
        Data generator for prepare input data.
        :param data_path: a path to the folder where the data is stored.
        :param json_name: the name of the json file that contains information about the files to download.
        :param augmentation_data: if this parameter is True, then augmentation is applied to the training dataset.
        :param image_shape: this is image shape (channels, height, width).
        :param is_train: if is_train = True, then we work with train images, otherwise with test.
        :param shuffle_data: if this parameter is True, then data will be shuffled every time.
        """

        self.image_shape = image_shape
        self.mean_std = mean_std

        # read json
        with open(os.path.join(data_path, json_name), 'r') as f:
            self.index = json.load(f)

        # augmentation data
        if is_train == 'train':
            self.data = [y for x in os.walk(os.path.join(data_path, 'train')) for y in glob(os.path.join(x[0], '*.jpg'))]
            augmentation = self.augmentation_images(augmentation_data)
        elif is_train == 'val':
            self.data = [y for x in os.walk(os.path.join(data_path, 'val')) for y in glob(os.path.join(x[0], '*.jpg'))]
            augmentation = self.augmentation_images()
        else:
            self.data = [y for x in os.walk(os.path.join(data_path, 'test')) for y in glob(os.path.join(x[0], '*.jpg'))]
            augmentation = self.augmentation_images()

        self.aug = augmentation
        if shuffle_data:
            self.on_epoch_end()

    def on_epoch_end(self) -> None:
        """
        Random shuffling of data at the end of each epoch.
        """
        np.random.shuffle(self.data)

    def __len__(self) -> int:
        return len(self.data)

    def prepare_image(self, img_path: str):

        lable = self.index[img_path.split('/')[-2]]

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = self.aug(image=img)
        img = img['image']

        img_tensor = torch.from_numpy(img)
        img_tensor = img_tensor.permute(2, 0, 1)
        img_tensor = img_tensor / 255.0

        return img_tensor, lable

    def __getitem__(self, idx) -> Tuple[torch.tensor, torch.tensor]:
        """
        This function prepares the image and label.
        :param idx: a number of the element to load.
        :return: image tensor and label tensor.
        """

        img_path = self.data[idx]
        img_tensor, lable = self.prepare_image(img_path=img_path)
        lable = torch.tensor(lable)

        return img_tensor, lable

    def augmentation_images(self, augm: bool = False) -> A.Compose:
        """
        This function performs data augmentation.
        :return: augment data
        """
        if augm is True:
            aug = A.Compose([
                A.Normalize(mean=(self.mean_std[0]), std=(self.mean_std[1])),
                A.Resize(height=self.image_shape[2], width=self.image_shape[1]),
                A.Blur(blur_limit=4.0, p=0.3),
                A.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.2, always_apply=False, p=0.3),
                A.GaussNoise(var_limit=(10.0, 100.0), mean=0, always_apply=False, p=0.3),
                A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=5, val_shift_limit=5, always_apply=True,
                                     p=0.4),
                A.RandomRotate90(p=1.0)
            ])
        else:
            aug = A.Compose([A.Normalize(mean=(self.mean_std[0]), std=(self.mean_std[1])),
                             A.Resize(height=self.image_shape[2], width=self.image_shape[1])])

        return aug


if __name__ == '__main__':
    r = CustomDataset('../data', 'index.json')
    r[2]
    print()