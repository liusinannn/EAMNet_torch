import argparse
import logging
import os
import sys
import math
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lr_scheduler
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
from dataset_3task import BasicDataset

import matplotlib.pyplot as plt

from EAMNet import *
if torch.cuda.device_count() > 1:
    print("Let's use {0} GPUs!".format(torch.cuda.device_count()))


class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        bce = F.binary_cross_entropy_with_logits(input, target)
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.size(0)
        input = input.view(num, -1)
        target = target.view(num, -1)
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5*bce+dice


os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'



dir_img = r'\\'
dir_mask1 = r'\\'
dir_mask2 = r'\\'
dir_mask3 = r'\\'





dir_checkpoint = 'ckpts/'

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-g', '--gpu_id', dest='gpu_id', metavar='G', type=int, default=0, help='GPU ID')
    parser.add_argument('-u', '--unet_type', dest='unet_type', metavar='U', type=str, default='Mednet', help='UNet type: v1/v2/v3/DDNet')
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=150, help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2, help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001, help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False, help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=1, help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10, help='Percent of the data that is used as validation (0-100)')
    return parser.parse_args()


def train_net(unet_type, net, device, epochs=50, batch_size=1, lr=0.1, val_percent=0.1, save_cp=True, img_scale=1):
    dataset = BasicDataset(unet_type, dir_img, dir_mask1, dir_mask2,dir_mask3,img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val

    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)

    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        UNet type:       {unet_type}
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Dataset size:    {len(dataset)}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}''')

    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8)
    scaler = GradScaler()

    lf = lambda x: (((1 + math.cos(x * math.pi / epochs)) / 2) ** 1.0) * 0.95 + 0.05
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    criterion1 = BCEDiceLoss()
    criterion2 = nn.MSELoss()
    criterion3 = nn.BCEWithLogitsLoss()

    train_losses = []
    val_losses = []

    for epoch in range(epochs):
        net.train()
        total_train_loss = 0

        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks1 = batch['mask1']
                true_masks2 = batch['mask2']
                true_masks3 = batch['mask3']

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32
                true_masks1 = true_masks1.to(device=device, dtype=mask_type)
                true_masks2 = true_masks2.to(device=device, dtype=mask_type)
                true_masks3 = true_masks3.to(device=device, dtype=mask_type)

                optimizer.zero_grad()

                # with autocast():
                masks_pred1,masks_pred2,masks_pred3 = net(imgs)
                loss1 = criterion1(masks_pred1,true_masks1)
                loss2 = criterion2(masks_pred2,true_masks2)
                loss3_fuse = criterion3(masks_pred3, true_masks3)
                loss3 =loss3_fuse

                loss = (loss1 + loss2 + loss3)/1

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                pbar.update(imgs.shape[0])
                global_step += 1
                total_train_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)


        train_loss = total_train_loss / len(train_loader)
        train_losses.append(train_loss)

        net.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                imgs = batch['image'].to(device=device, dtype=torch.float32)
                true_masks1 = batch['mask1'].to(device=device, dtype=mask_type)
                true_masks2 = batch['mask2'].to(device=device, dtype=mask_type)
                true_masks3 = batch['mask3'].to(device=device, dtype=mask_type)


                # with autocast():
                masks_pred1, masks_pred2, masks_pred3 = net(imgs)
                loss1 = criterion1(masks_pred1, true_masks1)
                loss2 = criterion2(masks_pred2, true_masks2)
                loss3_fuse = criterion3(masks_pred3, true_masks3)
                loss3 =loss3_fuse
                loss = (loss1 + loss2 + loss3)/1

                total_val_loss += loss.item()

        val_loss = total_val_loss / len(val_loader)
        val_losses.append(val_loss)
        writer.add_scalar('Loss/validation', val_loss, global_step)

        logging.info(f'Epoch {epoch + 1} finished! Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')

        if save_cp and (epoch+1)%5==0:
            try:
                os.makedirs(dir_checkpoint, exist_ok=True)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(), os.path.join(dir_checkpoint, f'CP_epoch{epoch + 1}_SD_EAMNet_noFEM.pth'))
            logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_id)
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    if args.unet_type == 'Mednet':
        net = EAMNet()

    net.to(device=device)

    if args.load:
        net.load_state_dict(torch.load(args.load, map_location=device))
        logging.info(f'Model loaded from {args.load}')

    try:
        train_net(unet_type=args.unet_type, net=net, epochs=args.epochs, batch_size=args.batchsize,
                  lr=args.lr, device=device, img_scale=args.scale, val_percent=args.val / 100)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        sys.exit(0)
