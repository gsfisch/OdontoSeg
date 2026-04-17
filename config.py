import os
import torch

wandb_name = 'OdontoSeg_Reviewed'

training_config = {
  'experiment_name': 'vit_large_patch16_224_U-Net',
  'encoder' : 'vit_large_patch16_224',
  'architecture': 'U-Net',
  'epochs': 200,
  'batch_size': 8,
  'val_batch_size': 8, 
  'delay_per_batch': 1,
  'dataset_path' : '/home/fisch/Documents/OdontoSeg/datasets/Dataset_Imagens_Clinicas_V2.0',
  'loss_function' : 'dice',
  'optimizer': 'adamw',
  'scheduler_step_size': 0.8,
  'learning_rate': 1e-4,
  'weight_decay' : 1e-4,
  'classes': 4,
  'class_weigths': 
  [0.6471186223837316, 1.0, 2.613295558781593, 0.03166476775283598], # calculated by the inverse of pixels frequency
  #'library': 'torchseg', # smp and torchseg for now
  'library': 'torchseg',
  'encoder_depth': 4,
  'decoder_channels': (256, 128, 64, 32),
  'encoder_params': {
                'img_size': 512,
                "scale_factors": (8, 4, 2, 1),
            },
  'head_upsampling': 1,
  'freeze_encoder': True,
}

path_models = '/home/fisch/Documents/OdontoSeg/models/'
path_save_evaluation = '/home/fisch/Documents/OdontoSeg/experiments/comparison/'
path_save_evaluation_percentage = '/home/fisch/Documents/OdontoSeg/experiments/comparison-percentage/'
save_percentage = True

classes_color = [
  torch.tensor([255, 0, 0]),   # carcinoma
  torch.tensor([0, 255, 0]),   # leucoplasia
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
