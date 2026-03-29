import os
import cv2
import numpy as np
import torch.nn as nn
import warnings
import torch
import imageio.core.util
import segmentation_models_pytorch as smp
from torch.serialization import SourceChangeWarning
from util.blend import blendImage, predictAndBlendImage
from util.evaluation import predictImage
from util.model import make_model
from albumentations import Normalize
from torch.autograd import Variable
import ast


# Verificação de disponibilidade da GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

#DATASET_PATH = '/home/gioam/SSD/COMPARISON_RESIZE_CROP/resize/image'
DATASET_PATH = '/home/fisch/Documents/OdontoSeg/dataset_sri_lanka/validation/images'
#MODEL_PATH = '/home/gioam/projects/new-odonto-segmentation/experiments_4c/unet_resnet34_4c_dice_20250120-164901/best_model_by_mIoU_epoch_494.pth'
model_directory_path = '/home/fisch/Documents/OdontoSeg/models/swin_large_patch4_window7_224_MAnet'
#model_path = '/home/fisch/Documents/OdontoSeg/models/swin_large_patch4_window7_224_MAnet/swin_large_patch4_window7_224_MAnet.pth'
model_file_name = 'swin_large_patch4_window7_224_MAnet.pth'
configs_file_name = "config.txt"
#ENCODER = 'resnet34'
#DECODER = 'U-Net'
SAVE_IMAGES_PATH = '/home/fisch/Documents/OdontoSeg/evaluation_results'
CLASSES = 4

# init model

folder = os.path.join(DATASET_PATH)
experiment_paths = SAVE_IMAGES_PATH
normalize = False
percentage = True

# load pretrained model and the weights 
'''
print(f'Model initializing: ${MODEL_PATH}')
model = make_model(ENCODER, DECODER, classes=4).cuda()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()
'''

print(f'Model initializing: ${model_directory_path}')
#model = make_model(ENCODER, DECODER, classes=4).cuda()
#model.load_state_dict(torch.load(MODEL_PATH))
#model.eval()

# Read training configurations
training_config = {}
with open(os.path.join(model_directory_path, configs_file_name), 'r') as configs_file:
    training_config = ast.literal_eval(configs_file.read())


# Initialize and load model
model = make_model(training_config['encoder'], training_config['architecture'], 
    classes=training_config['classes'], library=training_config['library'],
    decoder_channels=training_config['decoder_channels'], encoder_depth=training_config['encoder_depth'],
    encoder_params=training_config['encoder_params'], head_upsampling=training_config['head_upsampling']).cuda()

model.load_state_dict(torch.load(os.path.join(model_directory_path, model_file_name), weights_only="True"))
model.eval()


for filename in os.listdir(folder):
    original_image_path = os.path.join(folder, filename)

    # predictImage(model, original_image_path, SAVE_IMAGES_PATH, filename)
    predictAndBlendImage(model, original_image_path, SAVE_IMAGES_PATH, filename)





'''
import os
import cv2
import numpy as np
import torch.nn as nn
import warnings
import torch
import imageio.core.util
import segmentation_models_pytorch as smp
from torch.serialization import SourceChangeWarning
from util.blend import blendImage, predictAndBlendImage
from util.evaluation import predictImage
from util.model import make_model
from albumentations import Normalize
from torch.autograd import Variable
# Verificação de disponibilidade da GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

DATASET_PATH = '/home/gioam/SSD/COMPARISON_RESIZE_CROP/resize/image'
MODEL_PATH = '/home/gioam/projects/new-odonto-segmentation/experiments_4c/unet_resnet34_4c_dice_20250120-164901/best_model_by_mIoU_epoch_494.pth'
ENCODER = 'resnet34'
DECODER = 'U-Net'
SAVE_IMAGES_PATH = '/home/gioam/SSD/COMPARISON_RESIZE_CROP/resize/result'
CLASSES = 4

# init model

folder = os.path.join(DATASET_PATH)
experiment_paths = SAVE_IMAGES_PATH
normalize = False
percentage = True

# load pretrained model and the weights 

print(f'Model initializing: ${MODEL_PATH}')
model = make_model(ENCODER, DECODER, classes=4).cuda()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

for filename in os.listdir(folder):
    original_image_path = os.path.join(folder, filename)

    # predictImage(model, original_image_path, SAVE_IMAGES_PATH, filename)
    predictAndBlendImage(model, original_image_path, SAVE_IMAGES_PATH, filename)


'''