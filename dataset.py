#-*- coding:utf-8 -*-

from torch.utils.data import Dataset
from torchvision.transforms import RandomCrop, Compose, ToPILImage, Resize, GaussianBlur
from PIL import Image
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


class ScriptImageGenerator(Dataset):
    def __init__(self, imagefolder, input_size, transform=None):
        self.imagefolder = imagefolder
        self.input_size = input_size
        self.inputfiles = glob(os.path.join(imagefolder, '*'))
        self.row_size = 400
        self.min_height = 300
        self.transform = transform

    def inverte(self, image:np.ndarray):
        return cv2.bitwise_not(image)

    def choose_img(self):
        """
          - TODO: choose by font?
        """
        return np.random.choice(self.inputfiles)

    def __len__(self):
        return len(self.inputfiles)

    def __getitem__(self, index):
        try:
            img = np.asarray(Image.open(self.choose_img()))
            img = self.inverte(img)
            min_height, min_width = img.shape[:2]
            if min_height < self.min_height:
                padding = np.zeros((self.min_height-min_height, min_width, 3), dtype=np.uint8, order='C')
                img = np.concatenate((img, padding), axis=0)
                min_height = self.min_height
            while (min_width < self.row_size):
                    picked_img = np.asarray(Image.open(self.choose_img()))
                    picked_img = self.inverte(picked_img)
                    h, w = picked_img.shape[:2]
                    min_height = min_height if min_height < h else h
                    if min_height < self.min_height:
                        padding = np.zeros((self.min_height-min_height, w, 3), dtype=np.uint8, order='C')
                        picked_img = np.concatenate((picked_img, padding), axis=0)
                        min_height = self.min_height
                    img = np.concatenate((img[:min_height, :, :], picked_img[:min_height, :, :]), axis=1)
                    min_height, min_width = img.shape[:2]

            if self.transform is not None:
                img = self.transform(img)
        except Exception as e:
            print("Error in Dataset:", e)
            return self[(index+1)%len(self)]
        return img

if __name__ == '__main__':
    transform = Compose([ToPILImage(), RandomCrop(200), Resize(128)])
    dataset = ScriptImageGenerator("../../Datasets/mn_ocr_synthetic/images", 128, transform=transform)
    img = dataset[0]
    plt.imshow(img)
    plt.show()
