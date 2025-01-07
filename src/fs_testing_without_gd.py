import argparse
from fs_configs import *
from fs_testing_with_gd import Flat_Unet_Tester

"""
Testing dataset without ground truth
"""

parser = argparse.ArgumentParser(description="Flat U-Net testing script")
parser.add_argument("--fits_files", type=str, default=g_testing_ha_fits,
                    help="original full-disk Ha solar images with fits files")
parser.add_argument("--filament_masks", type=str, default=g_testing_filament_masks, help="filament mask path")
parser.add_argument("--saved_path", type=str, default=g_checkpoint_path, help="saved path of checkpoint")
parser.add_argument("--save_result_path", type=str, default=g_save_visual_result_path,
                    help="save visual result")
parser.add_argument("--channels", type=int, default=g_channels, help="channels")
parser.add_argument("--size", type=tuple, default=g_img_size, help="input size")
parser.add_argument("--cuda", type=bool, default=g_cuda, help="cuda is available?")
parser.add_argument("--num_workers", type=int, default=g_num_workers)
parser.add_argument("--simp_list", type=list, default=g_simp_list, help="CSA_ConvBlocks or SCA_ConvBlocks?")


def start(args):
    """
    testing entry point (without ground truth)
    """
    flat_unet_tester = Flat_Unet_Tester(args.channels,
                                        args.fits_files,
                                        args.saved_path,
                                        args.size,
                                        args.cuda,
                                        args.num_workers,
                                        args.save_result_path,
                                        simp_list=args.simp_list,
                                        filament_masks=args.filament_masks)
    flat_unet_tester.go()


if __name__ == '__main__':
    start(parser.parse_args())
