import os
import torch

# ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152', 'resnext50_32x4d', 'resnext101_32x4d', 'resnext101_32x8d', 
# 'resnext101_32x16d', 'resnext101_32x32d', 'resnext101_32x48d', 'dpn68', 'dpn68b', 'dpn92', 'dpn98', 'dpn107', 'dpn131',
#  'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn', 'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn',
#  'senet154', 'se_resnet50', 'se_resnet101', 'se_resnet152', 'se_resnext50_32x4d', 'se_resnext101_32x4d', 
# 'densenet121', 'densenet169', 'densenet201', 'densenet161', 'inceptionresnetv2', 'inceptionv4', 'efficientnet-b0', 'efficientnet-b1', 'efficientnet-b2', 
# 'efficientnet-b3', 'efficientnet-b4', 'efficientnet-b5', 'efficientnet-b6', 'efficientnet-b7', 'mobilenet_v2', 'xception', 
# 'timm-efficientnet-b0', 'timm-efficientnet-b1', 'timm-efficientnet-b2', 'timm-efficientnet-b3', 'timm-efficientnet-b4', 
# 'timm-efficientnet-b5', 'timm-efficientnet-b6', 'timm-efficientnet-b7', 'timm-efficientnet-b8', 'timm-efficientnet-l2', 
# 'timm-tf_efficientnet_lite0', 'timm-tf_efficientnet_lite1', 'timm-tf_efficientnet_lite2', 'timm-tf_efficientnet_lite3', 
# 'timm-tf_efficientnet_lite4', 'timm-skresnet18', 'timm-skresnet34', 'timm-skresnext50_32x4d', 
# 'mit_b0', 'mit_b1', 'mit_b2', 'mit_b3', 'mit_b4', 'mit_b5', 'mobileone_s0', 'mobileone_s1', 'mobileone_s2', 'mobileone_s3', 'mobileone_s4']
wandb_name = 'OdontoSeg'

training_config = {
  'experiment_name': 'swin_base_patch4_window12_384_U-Net++',
  'encoder' : 'swin_base_patch4_window12_384',
  'architecture': 'U-Net++',
  'epochs': 200,
  'batch_size': 32,
  'val_batch_size': 32,
  'delay_per_batch': 5,
  'dataset_path' : '/home/fisch/Documents/OdontoSeg/dataset',
  'loss_function' : 'dice',
  'optimizer': 'adamw',
  'scheduler_step_size': 0.8,
  'learning_rate': 1e-4,
  'weight_decay' : 1e-4,
  'classes': 4,
  'class_weigths': [0.6471186223837316, 1.0, 2.613295558781593, 0.03166476775283598], # calculated by the inverse of pixels frequency
  
  
  
  'library': 'torchseg', # smp and torchseg for now
  'encoder_depth': 4,
  'decoder_channels': (256, 128, 64, 32),
  'encoder_params': {
                'img_size': 512,
                "scale_factors": (8, 4, 2, 1),
            },
  'head_upsampling': 2,
}

path_models = '/home/fisch/Documents/OdontoSeg/models/'
path_save_evaluation = '/home/fisch/Documents/OdontoSeg/experiments/comparison/'
path_save_evaluation_percentage = '/home/fisch/Documents/OdontoSeg/experiments/comparison-percentage/'
save_percentage = True

classes_color = [
  torch.tensor([255, 0, 0]),   # NMM
  torch.tensor([0, 255, 0]),   # DPMB
  torch.tensor([255, 255, 0]), # proliferativas
  torch.tensor([0, 0, 255])    # background
]

wandb_config ={
  **training_config,
  'data_augmentation': 'online',
  'train_samples': len(os.listdir(os.path.join(training_config['dataset_path'], 'train/images'))),
  'val_samples': len(os.listdir(os.path.join(training_config['dataset_path'], 'validation/images'))),
  'test_samples': len(os.listdir(os.path.join(training_config['dataset_path'], 'test/images')))
}

classes_color_float = [
  torch.FloatTensor([[[255, 0, 0]]]),   # NMM
  torch.FloatTensor([[[0, 255, 0]]]),   # DPMB
  torch.FloatTensor([[[255, 255, 0]]]), # proliferativas
  torch.FloatTensor([[[0, 0, 255]]])    # background
]
