import os
import torch


wandb_name = 'OdontoSeg'

validation_config = {
  'model_directory_path': 'models/efficientnet-b6_FPN',
  'configs_file_name': 'config.txt',
  'model_file_name': 'efficientnet-b6_FPN.pth',
  'epochs': 200,
   #'batch_size': 1,
   #'val_batch_size': 1,
  'delay_per_batch': 1,
  'dataset_path' : '/home/fisch/Documents/OdontoSeg/dataset',
  #'dataset_path' : '/home/fisch/Documents/OdontoSeg/dataset_sri_lanka',
  'loss_function' : 'dice',
  'optimizer': 'adamw',
  'scheduler_step_size': 0.8,
  'learning_rate': 1e-4,
  'weight_decay' : 1e-4,
  'classes': 4,
  'class_weigths': 
  [0.6471186223837316, 1.0, 2.613295558781593, 0.03166476775283598], # calculated by the inverse of pixels frequency
  #'library': 'torchseg', # smp and torchseg for now
  'library': 'smp',
  'encoder_depth': 4,
  'decoder_channels': (256, 128, 64, 32),
  'encoder_params': {
                'img_size': 512,
                "scale_factors": (8, 4, 2, 1),
                #"scale_factors": (16, 8, 4, 2),
                #"scale_factors": (4, 2, 1, 0.5),
            },
  'head_upsampling': 1,
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
  **validation_config,
  'data_augmentation': 'online',
  'train_samples': len(os.listdir(os.path.join(validation_config['dataset_path'], 'train/images'))),
  'val_samples': len(os.listdir(os.path.join(validation_config['dataset_path'], 'validation/images'))),
  'test_samples': len(os.listdir(os.path.join(validation_config['dataset_path'], 'test/images')))
}

classes_color_float = [
  torch.FloatTensor([[[255, 0, 0]]]),   # NMM
  torch.FloatTensor([[[0, 255, 0]]]),   # DPMB
  torch.FloatTensor([[[255, 255, 0]]]), # proliferativas
  torch.FloatTensor([[[0, 0, 255]]])    # background
]

