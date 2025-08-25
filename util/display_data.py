import torch
import wandb
from skimage.io import imread, imsave
import os
from albumentations import Normalize
from torch.autograd import Variable
import torch.nn as nn

mapping_tensor = [
  torch.tensor([[[0,   0,   255]]]),
  torch.tensor([[[0,   255,   0]]]),
  torch.tensor([[[255,   0,  0]]]),
]

def mask_to_class_torch(mask):
    just_torch = mask.copy()
    just_torch_arr = torch.from_numpy(just_torch).long().cuda()

    for index in range(len(mapping_tensor)):
        just_torch_arr[torch.where((just_torch_arr == mapping_tensor[index].cuda()).all(axis=2))] = index

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


def display_image(model, epoch):
    files = ['carc_22.png', 'carc_30819_1.png', 'carc_OM-255.png', 'leuc_2.png', 'leuc_IMG_2070.png', 'leuc_IMG_5700.png']
    folder = "./train-data/test/images/"
    softmax = nn.Softmax(dim=1).cuda()
    class_labels = {0: "background", 1: "leucoplasia", 2: "carcinoma"}

    for filename in files:

        original_image = imread(os.path.join(folder, filename))
        gt_mask = os.path.join(folder, filename).replace('images', 'masks')
        final_gt = mask_to_class_torch(imread(gt_mask))[:, :, 0].cpu().numpy()

        img_input = prepare_img(original_image.copy(), normalize=False)

        # predict output
        output = model(img_input)
        predicted_prob_values, predicted = torch.max(softmax(output), 1)
        final_output_new = predicted[0]
        final_output_new = final_output_new.cpu().numpy()

        wandb.log(
            {filename.replace('.png', '') : wandb.Image(original_image, masks={
                "predictions" : {
                    "mask_data" : final_output_new,
                    "class_labels" : class_labels
                },
                "ground_truth" : {
                    "mask_data" : final_gt,
                    "class_labels" : class_labels
                }
            })}, step=epoch)
