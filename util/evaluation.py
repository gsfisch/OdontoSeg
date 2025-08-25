import os
from albumentations import Normalize
import torch
from torch.autograd import Variable
from config import classes_color
import cv2
import numpy as np
from skimage.io import imread, imsave
import torch.nn as nn

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class_colors_dict = {i: color.float() for i, color in enumerate(classes_color)}
softmax = nn.Softmax(dim=1).to(device)

def prepare_img(img):
    """Prepara a imagem para entrada no modelo."""
    img = img / 255.0

    # Transforma para tensor e ajusta dimensões
    img_tensor = torch.FloatTensor(img).permute(2, 0, 1).unsqueeze(0).to(device)
    return Variable(img_tensor)

def mask_to_class(final_output):
    # Inicializa a máscara de resultado
    h, w = final_output.shape
    result = torch.zeros((h, w, 3), dtype=torch.float32)

    # Aplica as cores para cada classe
    for class_id, color in class_colors_dict.items():
        result[final_output == class_id] = color

    return  result.byte().cpu().numpy()

def calculate_percentage_area(probabilities, mask, image_path):
    """Calcula e exibe a área percentual de cada cluster."""
    
    blended_img = cv2.imread(image_path, )
    for class_id, color in class_colors_dict.items():
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

def predictImage(model, original_image_path, save_path, save_path_percentage, filename, percentage=True):
    with torch.no_grad():
        # read image
        img = imread(original_image_path)

        # prepare image
        img_input = prepare_img(img.copy())

        # predict output
        output = model(img_input)
        probabilities = softmax(output)
        predicted = torch.argmax(probabilities, dim=1)[0]
        # print(f'predicted_prob_values {predicted_prob_values[0][1][1]}')

        # TODO: metrics
        
        # Gera a máscara colorida
        final_img = mask_to_class(predicted)
        
        # save image predicted in the correct folder
        os.makedirs(save_path, exist_ok=True)
        imsave(save_path + '/' + str(filename), final_img)
            
        if(percentage):
            os.makedirs(save_path_percentage, exist_ok=True)
            image_percentage = os.path.join(save_path_percentage, str(filename))
            imsave(image_percentage, final_img)
            calculate_percentage_area(probabilities[0].cpu().numpy(), final_img, image_percentage)