import os
import cv2
import numpy as np
import torch.nn as nn
import warnings
import torch
import imageio.core.util
import segmentation_models_pytorch as smp
from torch.serialization import SourceChangeWarning
from skimage.io import imread, imsave
from albumentations import Normalize
from torch.autograd import Variable

cudaAvailable = torch.cuda.is_available()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

class_colors = {
    0: torch.tensor([0, 0, 255], dtype=torch.float32),  # background (blue)
    1: torch.tensor([0, 255, 0], dtype=torch.float32),  # leucoplasia (green)
    2: torch.tensor([0, 255, 255], dtype=torch.float32),  # leucoplasia (green)
    3: torch.tensor([255, 0, 0], dtype=torch.float32),  # carcinoma (red)
}

def predictAndBlendImage(model, original_image_path, save_images_path, filename):
    print(f"Prediction image: {filename}")
    softmax = nn.Softmax(dim=1).cuda()
    result_path = os.path.join(save_images_path, filename)
    img = imread(original_image_path)

    # prepare image
    img_input = prepare_img(img.copy())

    # predict output
    output = model(img_input)
    probabilities = softmax(output)
    predicted = torch.argmax(probabilities, dim=1)[0]
    # print(f'predicted_prob_values {predicted_prob_values[0][1][1]}')

    final_img = mask_to_class(predicted)
    print(result_path)
    print(final_img.shape)
    # save image predicted in the correct folder
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv2.imwrite(result_path, final_img)
    
    # blend image
    blendImage(original_image_path, result_path)
    calculate_percentage_area(probabilities[0].cpu().detach().numpy(), final_img, result_path)

def mask_to_class(final_output):
    # Inicializa a máscara de resultado
    h, w = final_output.shape
    result = torch.zeros((h, w, 3), dtype=torch.float32)

    # Aplica as cores para cada classe
    for class_id, color in class_colors.items():
        result[final_output == class_id] = color

    return result.cpu().numpy()

def prepare_img(img, normalize=False):
    """Prepara a imagem para entrada no modelo."""
    if normalize:
        aug = Normalize(p=1)
        img = aug(image=img)["image"]
    else:
        img = img / 255.0

    # Transforma para tensor e ajusta dimensões
    img_tensor = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return Variable(img_tensor)

def blendImage(original_path, image_path):
    """Cria uma imagem misturada entre a original e a máscara."""
    original_img = cv2.imread(original_path)
    mask  = cv2.imread(image_path)
    
    blue_color = [255, 0, 0]
    blue_mask = cv2.inRange(mask, np.array(blue_color), np.array(blue_color))
    non_blue_mask = cv2.bitwise_not(blue_mask)

    alpha = 0.6
    blended_image = cv2.addWeighted(original_img, alpha, mask, 1 - alpha, 0)
    blended_image = cv2.bitwise_and(blended_image, blended_image, mask=non_blue_mask)
    blended_image += cv2.bitwise_and(original_img, original_img, mask=blue_mask)

    cv2.imwrite(image_path, blended_image)

def calculate_percentage_area(probabilities, mask, image_path):
    """Calcula e exibe a área percentual de cada cluster."""
    
    blended_img = cv2.imread(image_path, )
    for class_id, color in class_colors.items():
        # ignorar background class 0
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
        print(probabilities)
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
        
