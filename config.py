import os
import torch

wandb_name = '3'

training_config = {
  'experiment_name': 'Transformer',
  'encoder' : 'resnet34',
  'architecture': 'U-Net',
  'epochs': 2,
  'batch_size': 2,
  'val_batch_size': 2,
  'dataset_path' : '/home/master/Documents/TCC/odonto_segmentation/dataset',
  'loss_function' : 'dice',
  'optimizer': 'rangerlars',
  'learning_rate': 1e-3, # 1e-5 e 1e-1
  'weight_decay' : 1e-4, # 
  'classes': 4,
  'class_weigths': [0.6471186223837316, 1.0, 2.613295558781593, 0.03166476775283598] # calculated by the inverse of pixels frequency
}

path_models = '/home/master/Documents/TCC/odonto_segmentation/models/'
path_save_evaluation = '/home/master/Documents/TCC/odonto_segmentation/experiments/comparison/'
path_save_evaluation_percentage = '/home/master/Documents/TCC/odonto_segmentation/experiments/comparison-percentage/'
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