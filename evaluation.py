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
import cv2

# Configurações
folder = os.path.join(training_config['dataset_path'], 'test/images')
experiment_name = 'unet_vgg19_4c_20241230-135806'
experiment_paths = os.path.join(path_models, experiment_name)
normalize = False
percentage = True

class_colors_list = {
    0: torch.tensor([255, 0, 0]),   # NMM
    1: torch.tensor([0, 255, 0]),   # DPMB
    2: torch.tensor([255, 255, 0]), # proliferativas
    3: torch.tensor([0, 0, 255])   # background
}

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
    # Inicializa a máscara de resultado
    h, w = final_output.shape
    result = torch.zeros((h, w, 3), dtype=torch.float32)

    # Aplica as cores para cada classe
    for class_id, color in class_colors_list.items():
        result[final_output == class_id] = color

    return result.cpu().numpy()

def format_iou_output(iou):
    return "[" + ",".join(f"{value:.4f}" for value in iou) + "]"


def ignore_warnings(*args, **kwargs):
    pass

def calculate_percentage_area(probabilities, mask, image_path):
    """Calcula e exibe a área percentual de cada cluster."""
    
    blended_img = cv2.imread(image_path, )
    for class_id, color in class_colors_list.items():
        # ignorar backg3ound class 0
        if class_id == 3:
            continue
        process_clusters(blended_img, mask, probabilities, class_id, tuple(color.tolist()))
    cv2.imwrite(image_path, blended_img)

def process_clusters(blended_img, mask, probabilities, probability_id, color, min_points=50):
    """Processa clusters de uma máscara binária, calcula médias e desenha informações."""
    mask_binary = cv2.inRange(mask, np.array(color), np.array(color))
    num_labels, labels = cv2.connectedComponents(mask_binary)
    clusters = {}

    # Identifica clusters que atendem ao número mínimo de pontos
    for label in range(1, num_labels):
        pixels = np.column_stack(np.where(labels == label))
        if len(pixels) >= min_points:
            clusters[label] = pixels

    # Processa e exibe informações dos clusters
    for label, pixels in clusters.items():
        mean_probability = np.mean(probabilities[probability_id, pixels[:, 0], pixels[:, 1]])
        centroid = np.mean(pixels, axis=0)

        print(f"Cluster {label} - Cor: {color}")
        print(f"Centroide: {centroid}")
        print(f"Média de probabilidade: {mean_probability:.2f}")
        print(f"Número de pontos: {pixels.shape[0]}")

        # Adiciona texto na imagem com blend
        cv2.putText(
            blended_img,
            f'{mean_probability * 100:.2f}%',
            (int(centroid[1]), int(centroid[0])),
            cv2.FONT_ITALIC,
            0.5,
            (0, 0, 0),
            2
        )


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
            probabilities = softmax(output)
            predicted = torch.argmax(probabilities, dim=1)

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
            
            save_path = os.path.join('./odonto-segmentation-comparison-4c/', experiment_name, os.path.basename(experiment_path))

            # save image predicted in the correct folder
            os.makedirs(save_path, exist_ok=True)
            imsave(save_path + '/' + str(filename), final_img)
            
            if(percentage):
                save_path_percentage = os.path.join('./odonto-segmentation-comparison-4c-percentage/', experiment_name, os.path.basename(experiment_path))
                os.makedirs(save_path_percentage, exist_ok=True)
                imsave(save_path_percentage + '/' + str(filename), final_img)
                calculate_percentage_area(probabilities, final_img, save_path_percentage)
