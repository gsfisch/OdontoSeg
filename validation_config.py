import os
import torch


wandb_name = 'OdontoSeg'

validation_config = {
  'model_directory_path': 'models/SwinUNETR',
  'configs_file_name': 'config.txt',
  'model_file_name': 'SwinUNETR.pth',
  'dataset_path' : '/home/fisch/Documents/OdontoSeg/datasets/Dataset_Imagens_Clinicas_V2.0',
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

