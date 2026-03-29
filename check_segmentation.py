import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
from util.data import get_data_generators
from util.model import make_model

from util.blend import predictAndBlendImage, blendImage, calculate_percentage_area

from loops import val_loop
from optimizers.main import optimizer
from datetime import datetime
from validation_config import validation_config
from torchinfo import summary
import torchseg
import ast
import cv2
import numpy as np
import matplotlib.pyplot as plt

'''
# Blend image with its mask
def blendImage(original_path, image_path):
    original_img = cv2.imread(original_path)
    mask  = cv2.imread(image_path)
    
    blue_color = [255, 0, 0]
    blue_mask = cv2.inRange(mask, np.array(blue_color), np.array(blue_color))
    non_blue_mask = cv2.bitwise_not(blue_mask)

    alpha = 0.8
    blended_image = cv2.addWeighted(original_img, alpha, mask, 1 - alpha, 0)
    blended_image = cv2.bitwise_and(blended_image, blended_image, mask=non_blue_mask)
    blended_image += cv2.bitwise_and(original_img, original_img, mask=blue_mask)

    return blended_image
    cv2.imwrite(image_path, blended_image)
'''

def check_segmentation():
    torch.cuda.empty_cache()
    dataset_path = '/home/fisch/Documents/OdontoSeg/dataset_sri_lanka_joint/'
    model_directory_path = 'models/swin_large_patch4_window7_224_MAnet'
    configs_file_name = 'config.txt'
    model_file_name = 'swin_large_patch4_window7_224_MAnet.pth'
    save_images_path =  os.path.join('segmentation_results/' + model_file_name[:-4])

    os.makedirs(save_images_path, exist_ok=False)

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
    summary(model, input_size=(validation_config['batch_size'], 3, 512, 512))


    for img_filename in os.listdir(dataset_path + 'images'):
        print(img_filename)
        img_path = dataset_path + 'images/' + img_filename
        mask_path = dataset_path + 'masks/' + img_filename

        #blended_img = blendImage(img_path, mask_path)

        #plt.imshow(blended_img)
        #plt.title(img_filename)
        #plt.show()

        predictAndBlendImage(model, img_path, save_images_path, img_filename)



    exit()

    
    # get data generators
    training_generator, valid_generator = get_data_generators()


    # Validate
    metrics_val = val_loop(valid_generator, model)


    # Print and log results
    print(  f'val_loss: {metrics_val["loss"]:.3f}\n' +
            f'val_acc: {metrics_val["accuracy"]:.3f}\n' +
            f'val_mIoU: {metrics_val["mIoU"]:.3f}\n' +
            f'val_dice: {metrics_val["dice"]:.3f}\n'
        )



if __name__ == "__main__":
    check_segmentation()
