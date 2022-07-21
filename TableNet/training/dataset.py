import torch

import torch.nn as nn
import pandas as pd
import numpy as np
from glob import glob
from sklearn.model_selection import train_test_split
from PIL import Image
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import tqdm
import config
import cv2
import os
import sys

class ImageFolder(nn.Module):
    def __init__(self, df, isTrain = True, transform = None):
        super(ImageFolder, self).__init__()
        self.df = df

        if transform is None:
            self.transform = A.Compose([
                A.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                    max_pixel_value=255,
                ),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        img_path, table_msak_path, col_mask_path = self.df.iloc[index, 0], self.df.iloc[index, 1], self.df.iloc[index, 2]
        image = np.array(Image.open("../" + img_path))
        table_image = torch.FloatTensor(np.array(Image.open("../" + table_msak_path))/255.0).reshape(1, 1024, 1024)
        column_image = torch.FloatTensor(np.array(Image.open("../" + col_mask_path)) / 255.0).reshape(1, 1024, 1024)

        image = self.transform(image = image)['image']

        return {"image": image, "table_image": table_image, "column_image": column_image}

    def get_mean_std(train_data, transform):
        dataset = ImageFolder(train_data, transform)
        train_loader = DataLoader(dataset, batch_size=128)

        mean = 0
        std = 0
        for img_dict in tqdm.tqdm(train_loader):
            batch_samples = img_dict["image"].size(0)
            images = img_dict["image"].view(batch_samples, img_dict["image"].size(1), -1)
            mean += images.mean(2).sum(0)
            std += images.std(2).sum(0)

        mean /= len(train_loader.dataset)
        std /= len(train_loader.dataset)

        print('mean', mean)
        print('std', std)

if __name__ == '__main__':

    df = pd.read_csv('../processed_data_v2.csv')
    dataset = ImageFolder(df[df['hasTable']==1])

    img_num = 0
    os.makedirs('dataset_module_test', exist_ok=True)
    for img_dict in dataset:
        save_image(img_dict["image"], f'dataset_module_test/image_{img_num}.png')
        save_image(img_dict["table_image"], f'dataset_module_test/table_image_{img_num}.png')
        save_image(img_dict["column_image"], f'dataset_module_test/column_image_{img_num}.png')

        img_num += 1

        if img_num == 6:
            break





