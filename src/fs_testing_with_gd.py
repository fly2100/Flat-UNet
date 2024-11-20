import time
import cv2
import numpy as np
import argparse
from torch.utils.data import DataLoader
import torch.nn as nn

from fs_configs import *

from fs_model import Flat_Unet
from fs_dataset import Halpha_Dataset

"""
Testing dataset with ground truth
"""

parser = argparse.ArgumentParser(description="Flat U-Net testing script for gd")
parser.add_argument("--fits_files", type=str, default=g_testing_gd_ha_fits,
                    help="original full-disk Ha solar images with fits format")
parser.add_argument("--filament_masks", type=str, default=g_testing_gd_filament_masks, help="filament mask path")
parser.add_argument("--saved_path", type=str, default=g_checkpoint_path, help="saved path of checkpoint")
parser.add_argument("--save_result_path", type=str, default=g_save_gd_visual_result_path,
                    help="save visual result")
parser.add_argument("--channels", type=int, default=g_channels, help="channels")
parser.add_argument("--size", type=tuple, default=g_img_size, help="input size")
parser.add_argument("--cuda", type=bool, default=g_cuda, help="cuda is available?")
parser.add_argument("--num_workers", type=int, default=g_num_workers)
parser.add_argument("--is_simp", type=bool, default=g_simp, help="simplified CSA_ConvBlocks")


def start(args):
    """
    Entry point for initiating the testing process.(with ground truth)

    Args:
        args (Namespace): Parsed command-line arguments containing configurations

    Initializes the Flat_Unet_Tester and begins the testing
    """
    flat_unet_tester = Flat_Unet_Tester(args.channels,
                                        args.fits_files,
                                        args.saved_path,
                                        args.size,
                                        args.cuda,
                                        args.num_workers,
                                        args.save_result_path,
                                        is_simp=args.is_simp,
                                        filament_masks=args.filament_masks)
    flat_unet_tester.go()


class Flat_Unet_Tester:
    def __init__(self,
                 channels,
                 ha_fits,
                 checkpoint_path,
                 img_size,
                 cuda,
                 num_workers,
                 save_result_path,
                 is_simp=True,
                 filament_masks=None,
                 ):
        self.channels = channels
        self.img_size = img_size
        self.is_simp = is_simp
        self.checkpoint_path = checkpoint_path
        self.device = torch.device("cuda" if cuda else "cpu")
        self.testing_dataset = Halpha_Dataset(ha_fits, img_size, filament_masks)
        self.save_result_path = save_result_path

        self.testing_loader_args = dict(shuffle=True, num_workers=num_workers) if cuda \
            else dict(shuffle=True, drop_last=True)

        self.testing_loader = self.dataloader()

    def predict(self, model, loader, loss_fn):
        print("Entered Evaluation")
        with torch.no_grad():
            model.eval()

            dumpy_input = torch.randn(1,1,self.img_size[0],self.img_size[1])
            dumpy_input = dumpy_input.to(self.device)
            model(dumpy_input)

            i = 1
            t = 0
            for x, y, filename in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                s = time.time()
                y_pred = model(x)
                g = round((time.time() - s) * 1000, 2)
                t += g
                print("Time-consuming: " + str(g) + " ms", end="  ")

                self.postprocess(i, filename, x, y, y_pred)
                i += 1
        return t

    def dataloader(self):
        return DataLoader(self.testing_dataset, **self.testing_loader_args)

    def postprocess(self, index, filename, x, y, y_pred):
        """
        save visual results(8bit)
        original, ground truth, predicted result
        """
        x = x.squeeze().cpu().numpy() * 255
        y = y.squeeze().cpu().numpy() * 255
        y = y.astype('uint8')

        y_pred_ = torch.sigmoid(y_pred.squeeze())
        y_pred_ = y_pred_ > 0.5
        y_pred_ = y_pred_.cpu().long().numpy() * 255
        y_pred_result = y_pred_.astype('uint8')

        split = np.ones((self.img_size[1], 1)) * 255

        combined_visual_data = np.hstack((x, split, y, split, y_pred_result))

        filepath = self.save_result_path + "_" + str(self.channels) + "/" + str(index) + ".png"
        save_img_with_check(combined_visual_data, filepath)

        single_filepath = self.save_result_path + "_" + str(self.channels) + "/single/" + filename[0] + ".png"
        save_img_with_check(y_pred_result, single_filepath)
        print(filepath)

    def go(self):
        loss_fn = nn.BCEWithLogitsLoss()
        model = Flat_Unet(flat_channels=self.channels, is_simp=self.is_simp)
        # model = build_unet()
        model.float()
        model.to(self.device)

        model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))
        model.eval()

        # test_loss, pred_masks = self.predict(model, self.testing_loader, loss_fn)
        t = self.predict(model, self.testing_loader, loss_fn)
        print("average time: " + str(t) + "ms")


def save_img_with_check(img, path):
    folder_path = os.path.dirname(path)

    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    cv2.imwrite(path, img)


if __name__ == '__main__':
    start(parser.parse_args())
