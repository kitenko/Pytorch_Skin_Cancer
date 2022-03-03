import os
from glob import glob

import cv2
import torch
import torchvision.transforms as transforms
from tqdm import tqdm
from torch.utils.data import DataLoader, Dataset

from config import PATH_DATA, BATCH_SIZE


class GetData(Dataset):
    def __init__(self, data_path: str = PATH_DATA, is_train: str = 'val'):
        """
        Data generator for prepare input data.
        :param data_path: a path to the folder where the data is stored.
        :param is_train: if is_train = True, then we work with train images, otherwise with test.
        """

        self.convert_to_tensor = transforms.ToTensor()

        if is_train == 'train':
            self.data = [y for x in os.walk(os.path.join(data_path, 'train')) for y in glob(os.path.join(x[0], '*.jpg'))]
        elif is_train == 'val':
            self.data = [y for x in os.walk(os.path.join(data_path, 'val')) for y in glob(os.path.join(x[0], '*.jpg'))]
        else:
            self.data = [y for x in os.walk(os.path.join(data_path, 'test')) for y in glob(os.path.join(x[0], '*.jpg'))]

    def __len__(self) -> int:
        return len(self.data)

    def prepare_image(self, img_path: str):

        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img_tensor = self.convert_to_tensor(img)

        return img_tensor

    def __getitem__(self, idx) -> torch.tensor:
        """
        This function prepares the image and label.
        :param idx: a number of the element to load.
        :return: image tensor.
        """

        img_path = self.data[idx]
        img_tensor = self.prepare_image(img_path=img_path)

        return img_tensor


def count_mean_std_for_image(dataloader):
    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for data in tqdm(dataloader):
        channels_sum += torch.mean(data, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(data ** 2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches
    std = (channels_squared_sum / num_batches - mean ** 2) ** 0.5

    return mean, std


train_data = GetData(data_path='../data', is_train='train')
train_dataloader = DataLoader(dataset=train_data, batch_size=BATCH_SIZE, drop_last=True)


if __name__ == '__main__':
    print(count_mean_std_for_image(train_dataloader))
