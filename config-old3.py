import os
import torch

wandb_name = 'odonto_segmentation'

training_config = {
  'experiment_name': 'unet_vgg19_ce_loss_test3',
  'encoder' : 'vgg19',
  'architecture': 'U-Net',
  'epochs': 500,
  'batch_size': 8,
  'val_batch_size': 8,
  'dataset_path' : './odonnto-segmentation-train-data',
  'loss_function' : 'ce',
  'optimizer': 'adam',
  'learning_rate': 1e-6, # 1e-5 e 1e-1
  'weight_decay' : 1e-2, # 
  'classes': 3,
  'class_weigths': [0.14647541984259294, 1.4230167810174439, 1] # calculated by the inverse of pixels frequency
}

path_models = '/home/gioam/projects/new-odonto-segmentation/new_model_experiments/'

classes_color = [
  torch.tensor([[[0,   0,   255]]]),  # background id 0
  torch.tensor([[[0,   255,   0]]]),  # leucoplasia id 1
  torch.tensor([[[255,   0,  0]]]),   # carcinoma id 2
]

wandb_config ={
  **training_config,
  'data_augmentation': 'offline',
  'train_samples': len(os.listdir(os.path.join(training_config['dataset_path'], 'train/images'))),
  'val_samples': len(os.listdir(os.path.join(training_config['dataset_path'], 'validation/images'))),
  'test_samples': len(os.listdir(os.path.join(training_config['dataset_path'], 'test/images')))
}