#!/usr/bin/env python
# coding: utf-8

# In[]:

###########################  IMPORTS   ############################################# 

import torch
import torch.nn as nn
import torch.nn.functional as F
import monai
from monai.networks.nets import UNet,VNet,DynUNet,UNet_ENN
from monai.networks.utils import one_hot
from monai.transforms import (
    AsDiscrete,
    AddChanneld,
    AsChannelFirstd,
    Compose,
    LoadNiftid,
    RandCropByPosNegLabeld,
    RandRotate90d,
    ScaleIntensityd,
    ToTensord,
)
from monai.visualize import plot_2d_or_3d_image
from monai.data.utils import list_data_collate, worker_init_fn
from monai.inferers import sliding_window_inference
from monai.metrics import DiceMetric
from monai.metrics import compute_meandice
from torch.autograd import Variable
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.backends.cudnn as cudnn
import torchvision
from torchvision import datasets, models, transforms
import csv
import time
import SimpleITK as sitk
from os.path import splitext,basename
import random
from glob import glob

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from copy import copy
import os
import numpy as np
from torch.utils.tensorboard import SummaryWriter

#from global_tools.tools import display_loading_bar
from class_modalities.transforms import LoadNifti, Roi2Mask, ResampleReshapeAlign, Sitk2Numpy, ConcatModality
from monai.utils import first, set_determinism


train_transforms = Compose(
    [  # read img + meta info
        LoadNifti(keys=["pet_img", "ct_img", "mask_img"]),
        Sitk2Numpy(keys=['pet_img', 'ct_img', 'mask_img']),
        # user can also add other random transforms

        # Prepare for neural network
        ConcatModality(keys=['pet_img', 'ct_img']),
        AddChanneld(keys=["mask_img"]),  # Add channel to the first axis
        ToTensord(keys=["image", "mask_img"]),
    ])
# without data augmentation for validation
val_transforms = Compose(
    [  # read img + meta info
        LoadNifti(keys=["pet_img", "ct_img", "mask_img"]),
        Sitk2Numpy(keys=['pet_img', 'ct_img', 'mask_img']),
        # Prepare for neural network
        ConcatModality(keys=['pet_img', 'ct_img']),
        AddChanneld(keys=["mask_img"]),  # Add channel to the first axis
        ToTensord(keys=["image", "mask_img"]),
    ])


device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

base_path="/home/lab/hualing/2.5_SUV_dilation" #####the path you put the pre_processed data.
pet_path = base_path + '/' + 'pet_test'
ct_path = base_path + '/' + 'ct_test'
mask_path = base_path + '/' + 'pet_test_mask'
PET_ids = sorted(glob(os.path.join(pet_path, '*pet.nii')))
CT_ids = sorted(glob(os.path.join(ct_path, '*ct.nii')))
MASK_ids = sorted(glob(os.path.join(mask_path, '*mask.nii')))

data_dicts= zip(PET_ids, CT_ids, MASK_ids)
files=list(data_dicts)


train_files = [{"pet_img": PET, "ct_img": CT, 'mask_img': MASK} for  PET, CT, MASK in files[:138]]
val_files = [{"pet_img": PET, "ct_img": CT, 'mask_img': MASK} for  PET, CT, MASK in files[138:156]]
test_files = [{"pet_img": PET, "ct_img": CT, 'mask_img': MASK} for  PET, CT, MASK in files[156:]]


train_ds = monai.data.Dataset(data=train_files,transform=train_transforms)
val_ds = monai.data.Dataset(data=val_files,transform=val_transforms)
test_ds = monai.data.Dataset(data=test_files,transform=val_transforms)

train_loader = DataLoader(
        train_ds,
        batch_size=3,
        shuffle=True,
        num_workers=4,
        collate_fn=list_data_collate,
        pin_memory=torch.cuda.is_available(),)

val_loader = DataLoader(val_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)
test_loader = DataLoader(test_ds, batch_size=1, num_workers=4, collate_fn=list_data_collate)



trained_model_path="./pre-trained_model/best_metric_model_unet.pth"      ######path to the pretrained UNet model
model = UNet_ENN(
        dimensions=3,  # 3D
        in_channels=2,
        out_channels=2,
        kernel_size=5,
        channels=(8,16, 32, 64,128),
        strides=(2, 2, 2, 2),
        num_res_units=2,).to(device)
model_dict = model.state_dict()
pre_dict = torch.load(trained_model_path)
pre_dict = {k: v for k, v in pre_dict.items() if k in model_dict}
model_dict.update(pre_dict)

model.load_state_dict(model_dict)


####code to make sure only the parameters from ENN are optimized

for name, param in model.named_parameters():
    if param.requires_grad==True:
        print(name)
####code to make sure only the parameters from ENN are optimized

params = filter(lambda p: p.requires_grad, model.parameters())
optimizer = torch.optim.Adam(params, 1e-2)
dice_metric = monai.metrics.DiceMetric( include_background=False,reduction="mean")
scheduler=torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,'min',patience=10)
loss_function = monai.losses.DiceLoss(include_background=False,softmax=False,squared_pred=True,to_onehot_y=True)


# TODO : generate a learning rate scheduler

val_interval = 1
best_metric = -1
best_metric_epoch = -1
epoch_loss_values = list()
metric_values = list()

writer = SummaryWriter()


for epoch in range(100):
    print("-" * 10)
    print(f"epoch {epoch + 1}/{100}")
    model.train()
    epoch_loss = 0
    step = 0
    for batch_data in train_loader:
        step += 1
        inputs, labels = batch_data["image"].to(device), batch_data["mask_img"].to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        output=outputs[:, :2, :, :, :]+0.5*outputs[:, 2, :, :, :].unsqueeze(1)
        loss=loss_function(output, labels)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        epoch_len = len(train_ds) // train_loader.batch_size
        print(f"{step}/{epoch_len}, train_loss: {loss.item():.4f}")
        writer.add_scalar("train_loss", loss.item(), epoch_len * epoch + step)
    epoch_loss /= step
    epoch_loss_values.append(epoch_loss)
    print(f"epoch {epoch + 1} average loss: {epoch_loss:.4f}")
    scheduler.step(epoch_loss)
    if (epoch + 1) % val_interval == 0:
        model.eval()
        with torch.no_grad():
            metric_sum = 0.0
            metric_count = 0

            for val_data in val_loader:
                val_images, val_labels = val_data["image"].to(device), val_data["mask_img"].to(device)
                output = model(val_images)
                val_outputs = output[:, :2, :, :, :]+0.5*output[:, 2, :, :, :].unsqueeze(1)
                value = dice_metric(y_pred=val_outputs, y=val_labels)
                metric_count += len(value)
                metric_sum += value.item() * len(value)

            metric = metric_sum / metric_count
            metric_values.append(metric)
            if metric > best_metric:
                best_metric = metric
                best_metric_epoch = epoch + 1
                torch.save(model.state_dict(), "ENN_best_metric_model_segmentation3d_dict.pth")
                print("saved new best metric model")
            print(
                "current epoch: {} current mean dice: {:.4f} best mean dice: {:.4f} at epoch {}".format(
                    epoch + 1, metric, best_metric, best_metric_epoch
                )
            )


print(f"train completed, best_metric: {best_metric:.4f} at epoch: {best_metric_epoch}")
writer.close()
