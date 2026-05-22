import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import ast
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchinfo import summary
from util.model import make_model
from util.blend import predictAndBlendImage
from skimage.io import imread, imsave
from skimage import transform
from datetime import datetime
import numpy as np
import imageio.v2 as imageio
import time
import matplotlib.pyplot as plt


class_colors_list = {
    0: torch.tensor([1.0, 0.0, 0.0]),   # NMM
    1: torch.tensor([0.0, 1.0, 0.0]),   # DPMB
    2: torch.tensor([1.0, 1.0, 0.0]), # proliferativas
    3: torch.tensor([0.0, 0.0, 1.0])   # background
}


def prepare_img(img):
    temp = []

    imgTorch = torch.FloatTensor(transform.resize(img, (512, 512), anti_aliasing=True))
    imgTorch = imgTorch.permute(2, 0, 1)
    temp.append(imgTorch.numpy())
    tempTorch = torch.FloatTensor(np.array(temp))
    img_output = Variable(tempTorch.cuda())

    return img_output


def mask_to_class(final_output):
    # Initialize masks
    h, w = final_output.shape
    result = torch.zeros((h, w, 3), dtype=torch.float32)

    # Apply each class' colour
    for class_id, color in class_colors_list.items():
        result[final_output == class_id] = color

    return result.cpu().numpy()


def fix_colours(masks_image):
    mask_is_green = np.all(masks_image == [0, 1, 0], axis=-1)
    mask_is_yellow = np.all(masks_image == [1, 1, 0], axis=-1)

    masks_image[mask_is_green] = [1, 1, 0]
    masks_image[mask_is_yellow] = [0, 1, 0]

    return masks_image



def main():
    model_name = 'SegFormer_mit_b3'
    model_directory_path = f"models/{model_name}"                         
    '''
    images_path = [  'carcinoma_37', 'carcinoma_31547_2',       
                    'leucoplasia_10', 'leucoplasia_N-103',                  # Choose images
                    'ploliferativas_IMG_2088', 'ploliferativas_IMG_2089'
    ]

    images_path = [  'carcinoma_31227_1', #'',
                    'leucoplasia_40', #'',                  
                    'ploliferativas_IMG_9914', #''
    ]

    images_path = [
        'ploliferativas_hiperplasia epitelial'
    ]
    '''

    split = 'test'
    dataset_name = 'dataset_sri_lanka_no_healthy' 
    images_path = f"./datasets/{dataset_name}/{split}/images/"
    inference_directory_path = os.path.join("./inference/", model_directory_path[7:]) #, datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")[:-3])
    inference_directory_path = os.path.join("./inference/dataset_sri_lanka/", model_directory_path[7:]) 
    configs_file_name = "config.txt"
    model_file_name = model_directory_path[7:] + ".pth"


    # Read training configurations
    training_config = {}
    with open(os.path.join(model_directory_path, configs_file_name), 'r') as configs_file:
        training_config = ast.literal_eval(configs_file.read())


    # Initialize and load model
    if training_config['library'] == 'smp':
        model = make_model(training_config['encoder'], training_config['architecture'],
                            training_config['classes'], library='smp',
                            decoder_channels=training_config['decoder_channels'], encoder_depth=training_config['encoder_depth'],
                            freeze_encoder=training_config['freeze_encoder'], need_wrapper=training_config['need_wrapper']).cuda()

    else:    
        model = make_model(training_config['encoder'], training_config['architecture'], 
                       classes=training_config['classes'], library=training_config['library'],
                       decoder_channels=training_config['decoder_channels'], encoder_depth=training_config['encoder_depth'],
                       encoder_params=training_config['encoder_params'], head_upsampling=training_config['head_upsampling'],
                       freeze_encoder=training_config['freeze_encoder'], need_wrapper=training_config['need_wrapper']).cuda()


    model.load_state_dict(torch.load(os.path.join(model_directory_path, model_file_name), weights_only="True"))
    model.eval()

    summary(model, input_size=(training_config['batch_size'], 3, 512, 512)) 

    cumulative_elapsed_time = 0.0
    # Inference
    with torch.no_grad():
        softmax = nn.Softmax(dim=1).cuda()

        for image_name in os.listdir(images_path):
            print(f'{image_name=}')
            image_path = f"./datasets/{dataset_name}/{split}/images/" + image_name
            mask_path = f"./datasets/{dataset_name}/{split}/masks/" + image_name


            original_image = imread(image_path).astype(np.float32) / 255.0
            original_mask = imread(mask_path).astype(np.float32) / 255.0

            start = time.perf_counter()
            logits = model(prepare_img(original_image))    # Output Shape: (B, C ,H ,W) -> (1, 4, 512, 512)
            output = softmax(logits)
            predicted = torch.argmax(output, dim=1)
            masks_image = mask_to_class(predicted[0])

            

            resized = transform.resize(original_image, (512, 512), anti_aliasing=True)

            # Fix colours
            masks_image_fixed_colours = fix_colours(masks_image)
            original_mask_fixed_colours = fix_colours(original_mask)

            #plt.imshow(masks_image_fixed_colours)
            #plt.show()

            background = (masks_image[:, :, 2] > 0.8) & \
                            (masks_image[:, :, 0] < 0.2) & \
                            (masks_image[:, :, 1] < 0.2)

            alpha = (~background).astype(float)
            alpha = alpha[:, :, None]


            # Blend
            segmented_image = resized * (1 - 0.3 * alpha) + masks_image_fixed_colours * (0.3 * alpha)

            # segmented_image = 0.8*transform.resize(original_image, (512, 512), anti_aliasing=True) + 0.2*masks_image_fixed_colours

            end = time.perf_counter()
            elapsed_time = end - start
            cumulative_elapsed_time += elapsed_time
            print(f"Elapsed time: {elapsed_time:.6f} seconds")

            # Save images
            os.makedirs(inference_directory_path, exist_ok=True)
            imageio.imwrite(os.path.join(inference_directory_path, f"{image_name[:-4]}_mask_{model_name}.png"), masks_image_fixed_colours.astype(np.uint8) * 255)
            imageio.imwrite(os.path.join(inference_directory_path, f"{image_name[:-4]}_GT_{model_name}.png"), (original_mask_fixed_colours * 255).astype(np.uint8))
            imageio.imwrite(os.path.join(inference_directory_path, f"{image_name[:-4]}_seg_{model_name}.png"), (segmented_image * 255).astype(np.uint8))

        print(f"\nInference saved at: {inference_directory_path}")

        mean_elapsed_time = cumulative_elapsed_time / 115.0
        print(f"Mean Elapsed time: {mean_elapsed_time:.6f} seconds")

            

if __name__ == "__main__":
    main()
