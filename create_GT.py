import os
from util.blend import predictAndBlendImage
from skimage.io import imread, imsave
from skimage import transform
from datetime import datetime
import numpy as np
import imageio.v2 as imageio
import time


image_save_path = '/home/fisch/Documents/OdontoSeg/datasets/Dataset_Imagens_Clinicas_V2.0/test/GT'
images_path = f'{image_save_path[:-3]}/images'
masks_path = f'{image_save_path[:-3]}/masks'

os.makedirs(image_save_path, exist_ok=False)

for image_name in os.listdir(images_path):
            print(f'{image_name=}')
            image_path = images_path + '/' + image_name
            mask_path = masks_path + '/' + image_name


            original_image = imread(image_path).astype(np.float32) / 255.0
            mask_image = imread(mask_path).astype(np.float32) / 255.0

                
            background = (mask_image[:, :, 2] > 0.8) & \
                            (mask_image[:, :, 0] < 0.2) & \
                            (mask_image[:, :, 1] < 0.2)

            alpha = (~background).astype(float)
            alpha = alpha[:, :, None]

            # Blend
            segmented_image = original_image * (1 - 0.3 * alpha) + mask_image * (0.3 * alpha)

            # Save images
            imageio.imwrite(os.path.join(image_save_path, image_name), (segmented_image * 255).astype(np.uint8))