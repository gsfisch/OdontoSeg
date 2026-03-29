from albumentations import (
    HorizontalFlip,
    VerticalFlip,    
    RandomRotate90,    
    Transpose,
    Compose,
    Transpose,
    OneOf,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    CLAHE,
    RandomBrightnessContrast,  
    RandomGamma,
    RandomCrop,
    ShiftScaleRotate,
    Blur,
    GaussNoise,
    Solarize,
    ISONoise,
    Equalize,
    Posterize,
    Normalize,
    ReplayCompose,
    GaussianBlur,
    RandomScale,
    RandomResizedCrop,
    OneOf,
    ChannelDropout,
    # FancyPCA,
    HueSaturationValue,
    RGBShift,
    CLAHE
)
import numpy as np
import torch
import cv2

  # AUGMENTATIONS_TRAIN =  ReplayCompose([
  #       HorizontalFlip(p=0.5),
  #       VerticalFlip(p=0.5),
  #       RandomRotate90(p=0.5),
  #       Transpose(p=0.5),
  #       # Solarize(p=0.5),
  #       # ISONoise(p=0.5),
  #       Equalize(p=0.5),
  #       Posterize(p=0.5),
  #       # OneOf([
  #       #   ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
  #       #   GridDistortion(p=0.5),
  #       #   OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)                  
  #       # ], p=0.8),
  #       RandomContrast(p=0.5),    
  #       RandomBrightness(p=0.5),    
  #       RandomGamma(p=0.5),
  #       # GaussNoise(p=0.5),
  #       # GaussianBlur(p=0.5),
  #       # RandomResizedCrop(512, 512, p=0.5),
  #       # ShiftScaleRotate(p=0.5, scale_limit=0, border_mode=cv2.BORDER_CONSTANT)
  #       # Normalize(mean=(0.4539007 , 0.37364626, 0.34778222),std=(0.2540404 , 0.19213188, 0.18562855), p=1)
  #       # RandomCrop(448, 448),
  #   ], p = 1)
def one_of():
  return OneOf([
                OneOf([HorizontalFlip(p=0.5), VerticalFlip(p=0.5), RandomRotate90(p=0.5), Transpose(p=0.5)]),
                OneOf([Equalize(p=0.5), Posterize(p=0.5), RandomBrightnessContrast(p=0.5), RandomGamma(p=0.5)]),
                OneOf([ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03), GridDistortion(p=0.5),OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)])])
                

# def create_augmentations():
#     AUGMENTATIONS_TRAIN =  ReplayCompose([
#         one_of(),
#         one_of(),
#         one_of(),
#         one_of(),
#         # RandomResizedCrop(512, 512)
#     ], p = 1)

#     AUGMENTATIONS_VALID = ReplayCompose([])

#     return AUGMENTATIONS_TRAIN, AUGMENTATIONS_VALID

# def create_augmentations():
#     AUGMENTATIONS_TRAIN =  ReplayCompose([
#         HorizontalFlip(p=1.0),
#         VerticalFlip(p=0.5),
#         RandomRotate90(p=0.5),
#         Transpose(p=0.5),
#         # # Solarize(p=0.5),
#         # # ISONoise(p=0.5),
#         Equalize(p=0.5),
#         #Posterize(p=0.5),
#         OneOf([
#             ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
#             GridDistortion(p=0.5),
#             OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)                  
#         ], p=0.8),  
#         RandomBrightnessContrast(p=0.5),    
#         RandomGamma(p=0.5),
#         # # GaussNoise(p=0.5),
#         GaussianBlur(p=0.5),
#         RandomResizedCrop(512, 512, p=0.5),
#         # # FancyPCA(p=0.5),
#         # ChannelDropout(p=0.5),
#         # HueSaturationValue(p=0.5), 
#         # CLAHE(p=0.5),
#         ShiftScaleRotate(p=0.5, scale_limit=0, border_mode=cv2.BORDER_CONSTANT),
#         # # FancyPCA(p=0.5),
#         # # ChannelDropout(p=0.5),
#         # # Normalize(mean=(0.4539007 , 0.37364626, 0.34778222),std=(0.2540404 , 0.19213188, 0.18562855), p=1)
#         #RandomCrop(352, 480),
#         # Normalize(p=1.0)
#     ], p = 1)

#     AUGMENTATIONS_VALID = ReplayCompose([Normalize(p=1.0)])
#     # Normalize(p=1.0)
#     return AUGMENTATIONS_TRAIN, AUGMENTATIONS_VALID

# def create_augmentations():
#     AUGMENTATIONS_TRAIN =  ReplayCompose([
#         HorizontalFlip(p=1.0),
#         VerticalFlip(p=0.5),
#         RandomRotate90(p=0.5),
#         Transpose(p=0.5),
#         # # Solarize(p=0.5),
#         # # ISONoise(p=0.5),
#         # Equalize(p=0.5),
#         # # Posterize(p=0.5),
#         OneOf([
#             ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
#             GridDistortion(p=0.5),
#             OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)                  
#         ], p=0.8),  
#         # RandomBrightnessContrast(p=0.5),    
#         # RandomGamma(p=0.5),
#         # # GaussNoise(p=0.5),
#         GaussianBlur(p=0.5),
#         RandomResizedCrop(512, 512, p=0.5),
#         # # FancyPCA(p=0.5),
#         # # ChannelDropout(p=0.5),
#         # HueSaturationValue(p=0.5), 
#         # CLAHE(p=0.5),
#         ShiftScaleRotate(p=0.5, scale_limit=0, border_mode=cv2.BORDER_CONSTANT),
#         # # FancyPCA(p=0.5),
#         # # ChannelDropout(p=0.5),
#         # # Normalize(mean=(0.4539007 , 0.37364626, 0.34778222),std=(0.2540404 , 0.19213188, 0.18562855), p=1)
#         #RandomCrop(352, 480),
#         # Normalize(p=1.0)
#     ], p = 1.0)

#     AUGMENTATIONS_VALID = ReplayCompose([Normalize(p=0.0)])
#     # Normalize(p=1.0)
#     return AUGMENTATIONS_TRAIN, AUGMENTATIONS_VALID

def create_augmentations():
    AUGMENTATIONS_TRAIN =  ReplayCompose([
        HorizontalFlip(p=0.5),
        VerticalFlip(p=0.5),
        RandomRotate90(p=0.5),
        Transpose(p=0.5),
        # # Solarize(p=0.5),
        # # ISONoise(p=0.5),
        Equalize(p=0.5),
        Posterize(p=0.5),
        # OneOf([
        #     ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
        #     # GridDistortion(p=0.5),
        #     OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
        # ], p=0.8),  
        RandomBrightnessContrast(p=0.5),    
        RandomGamma(p=0.5),
        # # GaussNoise(p=0.5),
        # GaussianBlur(p=0.5),
        # RandomResizedCrop(512, 512, p=0.5),
        # # FancyPCA(p=0.5),
        # # ChannelDropout(p=0.5),
        # # HueSaturationValue(p=0.5), 
        # # CLAHE(p=0.5),
        #ShiftScaleRotate(p=0.5, scale_limit=0, border_mode=cv2.BORDER_CONSTANT),
        # # FancyPCA(p=0.5),
        # # ChannelDropout(p=0.5),
        # # Normalize(mean=(0.4539007 , 0.37364626, 0.34778222),std=(0.2540404 , 0.19213188, 0.18562855), p=1)
        #RandomCrop(352, 480),
        # Normalize(p=1.0)
    ], p = 1.0)

    AUGMENTATIONS_VALID = ReplayCompose([
        # Normalize(p=0.0)
        ])
    # Normalize(p=1.0)
    return AUGMENTATIONS_TRAIN, AUGMENTATIONS_VALID

def get_predict_aug():

    AUGMENTATIONS_TEST = ReplayCompose([])

    return AUGMENTATIONS_TEST

def mixup_data(x, y, alpha=1.0, use_cuda=True):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
  
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index, :]
    
    return mixed_x, mixed_y