import os.path

import cv2
import torch
from PIL import Image
import numpy as np
from torch.utils.data import Dataset
from astropy.io import fits


def min_max_norm(ha_data):
    min_val = ha_data.min()
    max_val = ha_data.max()
    data = (ha_data - min_val) / (max_val - min_val)
    return data


class Halpha_Dataset(Dataset):
    def __init__(self,
                 fits_path,
                 fits_size,
                 filament_masks_path=None,
                 transform=None):
        self.fits_path = fits_path
        self.filament_masks_path = filament_masks_path
        self.n_samples = len(fits_path)

        self.fits_size = fits_size
        self.transform = transform

    def read_fits_data(self, fits_name):
        with fits.open(fits_name) as hdul:
            ha_data = hdul[0].data
        return ha_data

    def preprocess_fits(self, ha_data):
        ha_data = Image.fromarray(ha_data)
        ha_data = ha_data.resize(self.fits_size, Image.BICUBIC)
        ha_data = np.asarray(ha_data).astype('float32')

        ha_norm = min_max_norm(ha_data)
        had = np.flipud(ha_norm)
        had = np.ascontiguousarray(had)
        return had

    def read_png(self, png_name):
        return Image.open(png_name)

    def preprocess_png(self, png_data):
        png = png_data.resize(self.fits_size, resample=Image.NEAREST)
        png = np.asarray(png).astype('float32')
        return png

    def __getitem__(self, index):
        # Read H-alpha
        ha_data = self.read_fits_data(self.fits_path[index])
        ha_data = self.preprocess_fits(ha_data)

        # Reading filament mask
        if self.filament_masks_path:
            filament_mask = self.read_png(self.filament_masks_path[index])
            filament_mask = self.preprocess_png(filament_mask)

        if self.transform:
            trans = self.transform(image=ha_data, mask=filament_mask)
            ha_data = trans['image']
            filament_mask = trans['mask']
            filament_mask = np.expand_dims(filament_mask, 0)
        else:
            ha_data = np.expand_dims(ha_data, 0)
            ha_data = torch.from_numpy(ha_data)

            if self.filament_masks_path:
                filament_mask = np.expand_dims(filament_mask, 0)
                filament_mask = torch.from_numpy(filament_mask)
            else:
                filament_mask = np.zeros((ha_data.size(1), ha_data.size(2)), dtype=np.uint8)
                cv2.putText(filament_mask,
                            "NO DATA",
                            (ha_data.size(1) // 2 - 70, ha_data.size(2) // 2 - 40),
                            cv2.FONT_HERSHEY_COMPLEX, 1, [1], 2)
        return ha_data, filament_mask, os.path.splitext(os.path.basename(self.fits_path[index]))[0]

    def __len__(self):
        return self.n_samples
