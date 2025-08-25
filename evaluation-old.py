import os
import torch
import numpy as np
import pandas as pd
import tqdm
import warnings
from torch import nn
from torch.autograd import Variable
from skimage.io import imread, imsave
from albumentations import Normalize
from util.metrics import calculate_accuracy_eval, calculate_miou_eval
from util.model import make_model
from config import training_config, classes_color, path_models

# Configurações
folder = os.path.join(training_config['dataset_path'], 'test/images')
experiment_name = 'unet_vgg19_focal_loss_20240911-172826'
experiment_paths = os.path.join(path_models, experiment_name)
normalize = False

# Funções auxiliares
def mask_to_class_torch(mask):
    just_torch = mask.copy()
    just_torch_arr = torch.from_numpy(just_torch).long().cuda()

    for index in range(len(classes_color)):
        just_torch_arr[torch.where((just_torch_arr == classes_color[index].cuda()).all(axis=2))] = index

    return just_torch_arr

def prepare_img(img, normalize=False):
    temp = []
    if (normalize):
        aug = Normalize(p=1)
        normalized_img = aug(image=img)['image']
    else:
        normalized_img = img / 255.0

    imgTorch = torch.FloatTensor(normalized_img)
    imgTorch = imgTorch.permute(2, 0, 1)
    temp.append(imgTorch.numpy())
    tempTorch = torch.FloatTensor(temp)
    img_input = Variable(tempTorch.cuda())

    return img_input

def mask_to_class(final_output):
    h, w = final_output.shape
    result = torch.zeros((h, w, 3))

    normal = (final_output == 0)
    leucoplasia = (final_output == 1)
    carcinoma = (final_output == 2)

    result[normal] = torch.FloatTensor([[[0,   0,   255]]])
    result[leucoplasia] = torch.FloatTensor([[[0,   255,   0]]])
    result[carcinoma] = torch.FloatTensor([[[255,   0,   0]]])

    return result

def format_iou_output(iou):
    return "[" + ",".join(f"{value:.4f}" for value in iou) + "]"


def ignore_warnings(*args, **kwargs):
    pass

# Carregar modelo
for experiment in os.listdir(experiment_paths):
    experiment_path = os.path.join(experiment_paths, experiment)
    model = make_model(training_config['encoder'], training_config['architecture'], classes=training_config['classes']).cuda()
    model.load_state_dict(torch.load(experiment_path))
    model.eval()

    # Inicializar métricas
    softmax = nn.Softmax(dim=1).cuda()
    miou = torch.zeros(3).float().cuda()
    number = torch.zeros(3).float().cuda()

    # Processar imagens
    total_acc = 0.0
    total_iou = 0.0
    with torch.no_grad():
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            img = imread(img_path)
            
            gt_mask_path = img_path.replace('images', 'masks')
            final_gt = mask_to_class_torch(imread(gt_mask_path))[:, :, 0]

            img_input = prepare_img(img.copy(), normalize)
            output = model(img_input)
            predicted_prob_values, predicted = torch.max(softmax(output), 1)

            # calculate metrics
            iou, miou, number, curr_iou = calculate_miou_eval(predicted.clone(), final_gt.clone(), miou, number)
            acc = calculate_accuracy_eval(predicted.clone(), final_gt.clone())
            
            print(f'iou: {iou}   miou: {miou}')
            print(acc)

            # transform image to be saved
            final_output_new = predicted[0]
            final_img = mask_to_class(final_output_new)

            # Atualize as métricas
            total_acc += acc.item()
            total_iou += iou
            
            save_path = os.path.join('./odonto-segmentation-comparison/', experiment_name, os.path.basename(experiment_path))

            # save image predicted in the correct folder
            os.makedirs(save_path, exist_ok=True)
            imsave(save_path + '/' + str(filename), final_img)
