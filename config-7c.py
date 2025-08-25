import os
import torch

wandb_name = 'odonto_classification'

training_config = {
  'experiment_name': 'unet_vgg19_big_training_ce',
  'encoder' : 'vgg19',
  'architecture': 'U-Net',
  'epochs': 2000,
  'batch_size': 8,
  'val_batch_size': 8,
  'dataset_path' : './classification-train-data',
  'loss_function' : 'ce',
  'optimizer': 'adam',
  'learning_rate': 1e-6, # 1e-5 e 1e-1
  'weight_decay' : 1e-2, # 
  'classes': 7,
  'class_weigths': [0.13718157170137144, 0.4676509297795559, 1.0, 0.39336124145864315, 0.4893928297462766, 0.6763642648308442, 0.004736386847800795] # calculated by the inverse of pixels frequency
}

path_models = '/home/gioam/projects/new-odonto-segmentation/new_model_experiments_classification/'

classes_color = [
    torch.tensor([[[255, 0, 0]]]),   # NMM
    torch.tensor([[[0, 255, 0]]]),   # DPMB
    torch.tensor([[[255, 0, 255]]]), # NBM
    torch.tensor([[[0, 255, 255]]]), # planas
    torch.tensor([[[255, 255, 0]]]), # proliferativas
    torch.tensor([[[255, 100, 0]]]), # LEI
    torch.tensor([[[0, 0, 255]]])    # background
]

classes_color_float = [
    torch.FloatTensor([[[255, 0, 0]]]),   # NMM
    torch.FloatTensor([[[0, 255, 0]]]),   # DPMB
    torch.FloatTensor([[[255, 0, 255]]]), # NBM
    torch.FloatTensor([[[0, 255, 255]]]), # planas
    torch.FloatTensor([[[255, 255, 0]]]), # proliferativas
    torch.FloatTensor([[[255, 100, 0]]]), # LEI
    torch.FloatTensor([[[0, 0, 255]]])    # background
]

wandb_config ={
  **training_config,
  'data_augmentation': 'offline',
  'train_samples': len(os.listdir(os.path.join(training_config['dataset_path'], 'train/images'))),
  'val_samples': len(os.listdir(os.path.join(training_config['dataset_path'], 'validation/images'))),
  'test_samples': len(os.listdir(os.path.join(training_config['dataset_path'], 'test/images')))
}