import argparse
import time
import torch.nn as nn
from torch.utils.data import random_split, DataLoader

from fs_model import Flat_Unet
from fs_configs import *
from fs_loss import DiceBCELoss, DiceLoss, FocalLoss
from fs_dataset import Halpha_Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchsummary import summary

parser = argparse.ArgumentParser(description="Flat U-Net training script")
parser.add_argument("--fits_files", type=str, default=g_ha_fits,
                    help="original full-disk Ha solar images with fits format")
parser.add_argument("--filament_masks", type=str, default=g_filament_masks, help="filament mask path")
parser.add_argument("--save_path", type=str, default=g_checkpoint_path, help="save path of checkpoint")
parser.add_argument("--channels", type=int, default=g_channels, help="channels")
parser.add_argument("--size", type=tuple, default=g_img_size, help="input size")
parser.add_argument("--cuda", type=bool, default=g_cuda, help="cuda is available?")
parser.add_argument("--batch_size", type=int, default=g_batch_size)
parser.add_argument("--epochs", type=int, default=g_num_epochs)
parser.add_argument("--lr", type=int, default=g_lr, help="learning rate")
parser.add_argument("--is_simp", type=bool, default=True, help="simplified CSA_ConvBlocks")
parser.add_argument("--num_workers", type=int, default=g_num_workers)
parser.add_argument("--val_percent", type=float, default=g_val_percent, help="dataset split")
parser.add_argument("--is_summary", type=bool, default=g_simp, help="model summary")


def start(args):
    """
    Entry point for initiating the training process.

    Args:
        args (Namespace): Parsed command-line arguments containing configurations

    Initializes the Flat_Unet_Trainer and begins the training loop while collecting
    the training and validation loss for analysis.
    """
    train_loss = []
    val_loss = []
    for c in range(1,2):
        flat_unet_trainer = Flat_Unet_Trainer(args.channels,  # 2 ** (c + 3) args.channels
                                              args.fits_files,
                                              args.save_path,
                                              args.size,
                                              args.cuda,
                                              args.batch_size,
                                              args.num_workers,
                                              args.epochs,
                                              args.lr,
                                              is_simp=args.is_simp,
                                              is_summary=args.is_summary,
                                              filament_masks=args.filament_masks,
                                              val_percent=args.val_percent)
        t, v = flat_unet_trainer.go()
        train_loss.append(t)
        val_loss.append(v)
    print(train_loss)
    print(val_loss)


def epoch_time(start_time, end_time):
    """
    Calculate the elapsed time between two moments.

    Args:
        start_time (float): The start time in seconds.
        end_time (float): The end time in seconds.

    Returns:
        tuple: Elapsed time in minutes and seconds.
    """
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


class Flat_Unet_Trainer:
    """
        Trainer class for managing the training process of the Flat U-Net model.

        Attributes:
            channels (int): Number of input channels for the (sca)csa-ConvBlock.
            ha_fits (str): The H-alpha FITS images.
            checkpoint_path (str): Path to save model checkpoints.
            img_size (tuple): Input image size (height, width).
            cuda (bool): Flag to use GPU if available.
            batch_size (int): Number of images per training batch.
            num_workers (int): Number of workers.
            num_epochs (int): Total number of training epochs.
            lr (float): Learning rate for optimization.
            is_summary (bool): Flag to display a summary of the model architecture.
            filament_masks (str): The filament mask images.
            val_percent (float): Percentage of data used for validation.
        """
    def __init__(self,
                 channels,
                 ha_fits,
                 checkpoint_path,
                 img_size,
                 cuda,
                 batch_size,
                 num_workers,
                 num_epochs,
                 lr,
                 is_simp=True,
                 is_summary=False,
                 filament_masks=None,
                 val_percent=0.1):
        self.device = torch.device("cuda" if cuda else "cpu")
        self.checkpoint_path = checkpoint_path
        self.is_summary = is_summary
        self.img_size = img_size
        self.num_epochs = num_epochs
        self.lr = lr
        self.is_simp = is_simp

        self.channels = channels

        self.transform = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.5),
            ToTensorV2(),
        ])
        self.train_dataset = Halpha_Dataset(ha_fits,
                                            img_size,
                                            filament_masks,
                                            self.transform)
        n_val = int(len(self.train_dataset) * val_percent)
        n_train = len(self.train_dataset) - n_val
        self.train_set, self.val_set = random_split(self.train_dataset, [n_train, n_val],
                                                    generator=torch.Generator().manual_seed(0))
        self.test_set = self.val_set

        self.train_loader_args = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers) if cuda \
            else dict(shuffle=True, batch_size=batch_size)

        self.val_loader_args = dict(shuffle=True, batch_size=batch_size, num_workers=num_workers) if cuda \
            else dict(shuffle=True, batch_size=batch_size)

        self.test_loader_args = dict(shuffle=True, num_workers=num_workers) if cuda \
            else dict(shuffle=True, drop_last=True)

        self.train_loader, self.val_loader, self.test_loader = self.dataloader()

        data_str = f"Dataset Size:\nTrain: {len(self.train_set)} - Validation: {len(self.val_set)} - Test: {len(self.test_set)}\n"
        print(data_str)

    def train(self, model, loader, optimizer, loss_fn):
        epoch_loss = 0.0
        print("Entered Training")
        model.train()
        for x, y, _ in loader:
            x = x.to(self.device)
            y = y.to(self.device)

            optimizer.zero_grad()
            y_pred = model(x)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        epoch_loss = epoch_loss / len(loader)
        return epoch_loss

    def evaluate(self, model, loader, loss_fn):
        epoch_loss = 0.0
        print("Entered Evaluation")
        model.eval()
        with torch.no_grad():
            for x, y, _ in loader:
                x = x.to(self.device)
                y = y.to(self.device)

                y_pred = model(x)
                loss = loss_fn(y_pred, y)
                epoch_loss += loss.item()

            epoch_loss = epoch_loss / len(loader)
        return epoch_loss

    def go(self):
        train_loss_list = []
        val_loss_list = []

        best_valid_loss = float("inf")

        model = Flat_Unet(flat_channels=self.channels, is_simp=self.is_simp)
        # model = build_unet()
        model.float()
        model = model.to(self.device)

        if self.is_summary:
            summary(model, input_size=tuple([1]) + self.img_size, batch_size=1, device=str(self.device))

        optimizer = torch.optim.Adam(model.parameters(), lr=self.lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True)

        loss_fn = nn.BCEWithLogitsLoss()

        for epoch in range(self.num_epochs):
            start_time = time.time()

            train_loss = self.train(model, self.train_loader, optimizer, loss_fn)
            valid_loss = self.evaluate(model, self.val_loader, loss_fn)

            scheduler.step(valid_loss)
            # Saving the model
            if valid_loss < best_valid_loss:
                save_path = self.checkpoint_path.replace(".pth", "_" + str(self.channels) + ".pth")
                data_str = f"Valid loss improved from {best_valid_loss:2.4f} to {valid_loss:2.4f}. Saving checkpoint: {save_path}"
                print(data_str)

                best_valid_loss = valid_loss
                torch.save(model.state_dict(), save_path)

            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)

            data_str = f'Epoch: {epoch + 1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s\n'
            data_str += f'\tTrain Loss: {train_loss:.3f}\n'
            data_str += f'\t Val. Loss: {valid_loss:.3f}\n'
            print(data_str)
            print(f"channel: {self.channels}")

            train_loss_list.append(round(train_loss, 3))
            val_loss_list.append(round(valid_loss, 3))
        return train_loss_list, val_loss_list

    def dataloader(self):
        train_loader = DataLoader(self.train_set, **self.train_loader_args)
        val_loader = DataLoader(self.val_set, **self.val_loader_args)
        test_loader = DataLoader(self.test_set, **self.test_loader_args)
        return train_loader, val_loader, test_loader


if __name__ == '__main__':
    start(parser.parse_args())
