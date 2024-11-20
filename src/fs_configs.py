from glob import glob
import os
import torch

"""enable cuda"""
g_cuda = torch.cuda.is_available()


"""device"""
g_device = torch.device("cuda" if g_cuda else "cpu")


"""input size"""
g_img_size = (512, 512)


"""initial learning rate"""
g_lr = 1e-3


"""validation percent"""
g_val_percent = 0.1


"""num_workers"""
g_num_workers = os.cpu_count() if g_cuda else 0

"""is SCA enable?"""
g_simp = True

"""channels"""
g_channels = 32

"""batch size"""
g_batch_size = 2


"""epochs for training"""
g_num_epochs = 100


"""save path of checkpoint(float32)"""
g_checkpoint_path = "../model/simp_funet_cpkt.pth"
# g_checkpoint_path = "../model/test.pth"

"""save path of checkpoint(int8)"""
g_quant_checkpoint_path = "../model/mem_cpkt.pth"


"""training dataset(training and validation)"""
g_ha_fits = sorted(glob('/pan2/fdata/fs_dataset/data/train/fits/*'))
g_filament_masks = sorted(glob('/pan2/fdata/fs_dataset/data/train/filament_masks/*'))


"""testing dataset with ground truth"""
g_testing_gd_ha_fits = sorted(glob('/pan2/fdata/fs_dataset/data/test_gd/fits/*'))
g_testing_gd_filament_masks = sorted(glob('/pan2/fdata/fs_dataset/data/test_gd/filament_masks/*'))
g_save_gd_visual_result_path = "/pan2/fdata/fs_dataset/results/visual"


"""testing dataset without ground truth"""
g_testing_ha_fits = sorted(glob('/pan2/fdata/fs_dataset/data/test_no_gd/fits/*'))
g_testing_filament_masks = None
g_save_visual_result_path = "/pan2/fdata/fs_dataset/results_no_gd/visual"
