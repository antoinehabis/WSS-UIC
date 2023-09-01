import os
import numpy as np 
from  openslide import OpenSlide
import sys
import pandas as pd
import matplotlib.pyplot as plt
import torch
import tifffile
import cv2
from tqdm import tqdm
from PIL import Image
from torchvision.models import vgg16
from glob import glob


path_camelyon = '/home/ahabis/sshfs/CAMELYON'

path_slide_healthy_train = os.path.join(path_camelyon,'train/normal')
path_slide_tumor_train = os.path.join(path_camelyon,'train/tumor')
path_slide_tumor_test = os.path.join(path_camelyon,'test/tumor')
path_annotations_train = os.path.join(path_camelyon,'annotations')
path_annotations_test = os.path.join(path_camelyon,'test/annotations')
path_patches_scribbles_train = os.path.join(path_camelyon,'patches_scribbles_train')
path_patches_scribbles_test = os.path.join(path_camelyon,'patches_scribbles_test')
path_dataframe_train = os.path.join(path_camelyon, 'dataframe_train.csv')
path_dataframe_test = os.path.join(path_camelyon, 'dataframe_test.csv')


path_preds = '/home/ahabis/sshfs_zeus/CAMELYON'
path_patches_test = os.path.join(path_preds,'patches_test')
path_patches_mask_test = os.path.join(path_preds,'patches_masks')
path_prediction_features = os.path.join(path_preds,'features_predictions')
path_slide_true_masks = os.path.join(path_preds,'truemasks')
path_prediction_patches = os.path.join(path_preds, 'patchesprediction')
path_uncertainty_maps = os.path.join(path_preds, 'uncertainty_maps')
path_heatmaps = os.path.join(path_preds, 'heatmaps')

path_weights = os.path.join(path_preds,'weights')

percentage_scribbled_regions = 0.1
ov = 0.5
ps = 512
bs = 16



class VGG16(torch.nn.Module):

    def __init__(self,model):
        super(VGG16, self).__init__()
        
        self.vgg16 = model
        self.relu = torch.nn.ReLU()
        self.fc = torch.nn.Linear(in_features = 1000,out_features = 1).cuda()
        self.sigmoid = torch.nn.Sigmoid()
        
    
    def forward(self, x):
        x0 = self.vgg16(x) #### output size  4096
        x1 = self.relu(x0)
        x2 = self.fc(x1)  #### output size  1
        x3 = self.sigmoid(x2)

        return x3
    
model = VGG16(vgg16(pretrained=True)).cuda()