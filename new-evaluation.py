import os
import cv2
import numpy as np
import torch.nn as nn
import warnings
import torch
import imageio.core.util
import segmentation_models_pytorch as smp
from torch.serialization import SourceChangeWarning
from util.evaluation import predictImage
from util.model import make_model
from config import training_config, path_models, path_save_evaluation, path_save_evaluation, path_save_evaluation_percentage, save_percentage
from albumentations import Normalize
from torch.autograd import Variable
# Verificação de disponibilidade da GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# init model

folder = os.path.join(training_config['dataset_path'], 'test/images')
experiment_name = 'unet_resnet34_20250310-194650'
experiment_paths = os.path.join(path_models, experiment_name)
normalize = False
percentage = True

# load pretrained model and the weights 
for experiment in os.listdir(experiment_paths):
    experiment_path = os.path.join(experiment_paths, experiment)
    
    print(f'Model initializing: ${experiment_path}')
    model = make_model(training_config['encoder'], training_config['architecture'], classes=training_config['classes']).cuda()
    model.load_state_dict(torch.load(experiment_path))
    model.eval()
    
    for filename in os.listdir(folder):
        original_image_path = os.path.join(folder, filename)
        save_path = os.path.join(path_save_evaluation, experiment_name, os.path.basename(experiment_path))
        save_path_percentage = os.path.join(path_save_evaluation_percentage, experiment_name, os.path.basename(experiment_path))
        
        predictImage(model, original_image_path, save_path, save_path_percentage, filename, save_percentage)

