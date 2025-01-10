import argparse
import logging
import os
os.environ['PROJ_LIB'] = r'C:\\Anaconda\\envs\\pytorch\\lib\\site-packages\\pyproj\\proj_dir\\share\\proj'
import numpy as np
import torch
import torch.nn.functional as F
import torchsummary as summary
from PIL import Image
from osgeo import gdal
from torchvision import transforms


from dataset_3task import BasicDataset

Image.MAX_IMAGE_PIXELS = None
from EAMNet import *



def predict_img(unet_type, net, full_img, device, img_scale=1, out_threshold=0.5):
    net.eval()
    img = torch.from_numpy(BasicDataset.preprocess(unet_type, full_img, img_scale))
    img = img.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img)
        # if net.n_classes > 1:
        #     probs = F.softmax(output, dim=1)
        # else:
        probs = torch.sigmoid(output[0])

        probs = probs.squeeze(0)
        tf = transforms.Compose([transforms.ToPILImage(), transforms.Resize(full_img.size[1]),
                                 transforms.ToTensor()])
        probs = tf(probs.cpu())
        full_mask = probs.squeeze().cpu().numpy()
    return full_mask



def get_args():
    class Args:
        gpu_id = 0
        unet_type = 'Mednet'  # 'v1', 'v2', 'v3', 'UNet3Plus_DeepSup', 或 'UNet3Plus_DeepSup_CGM'
        model = 'ckpts/CP_epoch75_SD_EAMNet_noFEM.pth'
        input_folder = r'E:\Si-nanLiu\pythonproject1\dataset\xinjiang\XJ_289\XJ\test\img\\'
        # input_folder = r'E:\Si-nanLiu\pythonproject1\dataset\SC\sichuan\test\image\\'
        output_folder = r'E:\Si-nanLiu\pythonproject1\res\xiaorongshiyan\SD\EAMNet_noFEM\\'  # 指定输出图像的文件夹路径
        viz = True
        no_save = False
        scale = 1
    return Args()


def get_output_filenames(args):
    in_files = [os.path.join(args.input_folder, f) for f in os.listdir(args.input_folder) if f.endswith(('.jpg', '.png', '.tif'))]
    out_files = []

    for f in in_files:
        pathsplit = os.path.splitext(f)
        out_files.append('{}_OUT{}'.format(pathsplit[0], pathsplit[1]))

    return out_files


def mask_to_image(mask):
    return Image.fromarray((mask * 255).astype(np.uint8))


def get_georeference_info(input_folder):
    input_files = [f for f in os.listdir(input_folder) if f.endswith(('.jpg', '.png', '.tif'))]
    input_file = os.path.join(input_folder, input_files[0])  # 使用第一个输入文件获取空间参考信息
    input_dataset = gdal.Open(input_file)
    geotransform = input_dataset.GetGeoTransform()
    projection = input_dataset.GetProjection()
    return geotransform, projection


def save_mask_with_georeference(mask, output_path, geotransform, projection):
    driver = gdal.GetDriverByName("GTiff")
    output_dataset = driver.Create(output_path, mask.shape[1], mask.shape[0], 1, gdal.GDT_Byte)
    output_dataset.SetGeoTransform(geotransform)
    output_dataset.SetProjection(projection)
    output_band = output_dataset.GetRasterBand(1)
    output_band.WriteArray(mask)
    output_band.FlushCache()
    output_dataset = None


if __name__ == '__main__':
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')


    net = MEDNet()

    logging.info('Loading model {}'.format(args.model))
    net.to(device=device)
    net.load_state_dict(torch.load(args.model, map_location=device))
    logging.info('Model loaded !')

    in_files = [os.path.join(args.input_folder, f) for f in os.listdir(args.input_folder) if f.endswith(('.jpg', '.png', '.tif'))]
    for i, fn in enumerate(in_files):
        geotransform, projection = get_georeference_info(os.path.dirname(fn))
        logging.info('\nPredicting image {} ...'.format(fn))
        img = Image.open(fn)
        mask = predict_img(unet_type=args.unet_type, net=net, full_img=img, img_scale=args.scale, device=device)
        # mask = predict_large_image(unet_type=args.unet_type,net=net,full_img=img,crop_size=512,device=device)

        # 将预测结果映射到 0-255 范围
        mask = (mask * 255).astype(np.uint8)
        if not args.no_save:
            out_fn = os.path.join(args.output_folder, os.path.basename(fn))
            save_mask_with_georeference(mask, out_fn, geotransform, projection)
            logging.info('Mask saved to {}'.format(out_fn))

        if args.viz:
            logging.info('Visualizing results for image {}, close to continue ...'.format(fn))



