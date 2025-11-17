import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import ast
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchinfo import summary
from util.model import make_model
from skimage.io import imread, imsave
from skimage import transform
from datetime import datetime
import numpy as np
import imageio.v2 as imageio


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


def main():
    model_directory_path = "./models/swin_base_patch4_window7_224_U-Net++"  # Choose model
    image_path = "./dataset/test/images/carcinoma_37.png"                   # Choose image
    image_path = "./dataset/test/images/como-saber-cancer-de-boca-7-1024x576.webp"
    configs_file_name = "configs_used.txt"
    model_file_name = model_directory_path[9:] + ".pth"
    inference_directory_path = os.path.join("./inference/", model_directory_path[9:], datetime.now().strftime("%Y-%m-%d_%H:%M:%S.%f")[:-3])


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

    summary(model, input_size=(training_config['batch_size'], 3, 512, 512))


    # Inference
    with torch.no_grad():
        softmax = nn.Softmax(dim=1).cuda()
        original_image = imread(image_path).astype(np.float32) / 255.0


        logits = model(prepare_img(original_image))    # Output Shape: (B, C ,H ,W) -> (1, 4, 512, 512)
        output = softmax(logits)
        predicted = torch.argmax(output, dim=1)
        masks_image = mask_to_class(predicted[0])
        segmented_image = 0.8*transform.resize(original_image, (512, 512), anti_aliasing=True) + 0.2*masks_image


        # Save images
        os.makedirs(inference_directory_path, exist_ok=True)
        imageio.imwrite(os.path.join(inference_directory_path, "masks_image.jpeg"), masks_image.astype(np.uint8) * 255)
        imageio.imwrite(os.path.join(inference_directory_path, "original_image.jpeg"), (original_image * 255).astype(np.uint8))
        imageio.imwrite(os.path.join(inference_directory_path, "segmented_image.jpeg"), (segmented_image * 255).astype(np.uint8))

        print(f"\nInference saved at: {inference_directory_path}")


if __name__ == "__main__":
    main()
