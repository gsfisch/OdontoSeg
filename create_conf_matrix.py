import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
from util.data import get_data_generators
from util.model import make_model
from loops import val_loop_conf_matrix, test_loop
from optimizers.main import optimizer
import wandb
from datetime import datetime
from validation_config import validation_config
from torchinfo import summary
import torchseg
import ast
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


label_name = {
    'efficientnet-b6_FPN': 'EfficientNet + FPN',
    'efficientnet-b6_Linknet': 'EfficientNet + LinkNet',
    'efficientnet-b6_U-Net': 'EfficientNet + U-Net',
    'resnet34_U-Net': 'ResNet34 + U-Net',
    'resnet34_Linknet': 'ResNet34 + LinkNet',
    'resnet34_FPN': 'ResNet34 + FPN',
    'resnet101_U-Net': 'ResNet101 + U-Net',
    'resnet101_Linknet': 'ResNet101 + LinkNet',
    'resnet101_FPN': 'ResNet101 + FPN',
    'SegFormer_mit_b0': 'SegFormer b0',
    'SegFormer_mit_b1': 'SegFormer b1',
    'SegFormer_mit_b2': 'SegFormer b2',
    'SegFormer_mit_b3': 'SegFormer b3',
    'SegFormer_mit_b4': 'SegFormer b4',
    'SegFormer_mit_b5': 'SegFormer b5',
    'vgg16_FPN': 'VGG16 + FPN',
    'vgg16_Linknet': 'VGG16 + LinkNet',
    'vgg16_U-Net': 'VGG16 + U-Net',
    'vit_large_patch16_224_FPN': 'ViT + FPN',
    'vit_large_patch16_224_Linknet': 'ViT + LinkNet',
    'vit_large_patch16_224_MAnet': 'ViT + MA-Net',
    'vit_large_patch16_224_U-Net': 'ViT + U-Net',
    'vit_large_patch16_224_U-Net++': 'ViT + U-Net++',
    'deit3_base_patch16_224_FPN': 'DeiT 3 + FPN',
    'deit3_base_patch16_224_Linknet': 'DeiT 3 + LinkNet',
    'deit3_base_patch16_224_MAnet': 'DeiT 3 + MA-Net',
    'deit3_base_patch16_224_U-Net': 'DeiT 3 + U-Net',
    'deit3_base_patch16_224_U-Net++': 'DeiT 3 + U-Net++',
    'swin_large_patch4_window7_224_FPN': 'Swin Transf. + FPN',
    'swin_large_patch4_window7_224_Linknet': 'Swin Transf. + LinkNet',
    'swin_large_patch4_window7_224_MAnet': 'Swin Transf. + MA-Net',
    'swin_large_patch4_window7_224_U-Net': 'Swin Transf. + U-Net',
    'swin_large_patch4_window7_224_U-Net++': 'Swin Transf. + U-Net++',
    'caformer_b36_FPN': 'CAFormer + FPN',
    'caformer_b36_Linknet': 'CAFormer + LinkNet',
    'caformer_b36_MAnet': 'CAFormer + MA-Net',
    'caformer_b36_U-Net': 'CAFormer + U-Net',
    'caformer_b36_U-Net++': 'CAFormer + U-Net++',
    'inceptionv4_FPN': 'Inception V4 + FPN',
    'inceptionv4_Linknet': 'Inception V4 + LinkNet',
    'inceptionv4_U-Net': 'Inception V4 + U-Net',
    'resnet101_FPN': 'ResNet101 + FPN',
    'resnet101_Linknet': 'ResNet101 + LinkNet',
    'resnet101_U-Net': 'ResNet101 + U-Net',
    'resnet34_FPN': 'ResNet34 + FPN',
    'resnet34_Linknet': 'ResNet34 + LinkNet',
    'MedFormer': 'MedFormer',
    'TransUNet': 'TransUNet',
    'UNETR_64': 'UNETR',
    'SwinUNETR': 'SwinUNETR',
}


skip_names = ['old_dataset', 'resnet101_FPN_batch8_rangerlars', 'resnet101_U-Net_batch8_rangerlars', 'UNETR_old', 'resnet101_Linknet_batch8']


def generate_confusion_matrix():
    torch.cuda.empty_cache()

    for model_name in os.listdir('./models'):
        if model_name in skip_names:
            continue

        model_directory_path = f'models/{model_name}'
        configs_file_name = 'config.txt'
        model_file_name = f'{model_name}.pth'


        # Read training configurations
        training_config = {}
        with open(os.path.join(model_directory_path, configs_file_name), 'r') as configs_file:
            training_config = ast.literal_eval(configs_file.read())


        # Initialize and load model
        if training_config['library'] == 'smp':
            model = make_model(training_config['encoder'], training_config['architecture'],
                                training_config['classes'], library='smp',
                                decoder_channels=training_config['decoder_channels'], encoder_depth=training_config['encoder_depth'],
                                freeze_encoder=training_config['freeze_encoder'], need_wrapper=training_config['need_wrapper']).cuda()

        else:    
            model = make_model(training_config['encoder'], training_config['architecture'], 
                        classes=training_config['classes'], library=training_config['library'],
                        decoder_channels=training_config['decoder_channels'], encoder_depth=training_config['encoder_depth'],
                        encoder_params=training_config['encoder_params'], head_upsampling=training_config['head_upsampling'],
                        freeze_encoder=training_config['freeze_encoder'], need_wrapper=training_config['need_wrapper']).cuda()

        
        model.load_state_dict(torch.load(os.path.join(model_directory_path, model_file_name), weights_only="True"))

        #summary(model, input_size=(training_config['batch_size'], 3, 512, 512))
        
        # get data generators
        training_generator, valid_generator, test_generator = get_data_generators(training_config['batch_size'], dataset_path=validation_config['dataset_path'])

        with torch.no_grad():

            # Validate
            metrics_val, conf_matrix = val_loop_conf_matrix(test_generator, model)


            # Print and log results
            print(  f'val_loss: {metrics_val["loss"]:.4f}\n' +
                    f'val_acc: {metrics_val["accuracy"]:.4f}\n' +
                    f'val_mIoU: {metrics_val["mIoU"]:.4f}\n' +
                    f'val_dice: {metrics_val["dice"]:.4f}\n' +
                    f'val_precision: {metrics_val["precision"]:.4f}\n' +
                    f'val_recall: {metrics_val["recall"]:.4f}\n'
                )

            print(f'{conf_matrix=}')
            print(f'{conf_matrix.shape=}')
            print(f'{conf_matrix.dtype=}')

            cm = conf_matrix.detach().cpu().numpy()

            # Optional: convert to integers for display (confusion matrices are counts)
            #cm = np.log1p(cm)

            cm = cm / cm.sum(axis=1, keepdims=True)

            #cm = cm.astype(int)

            #annot = np.vectorize(lambda x: f"{x:,}".replace(",", "."))(cm)

            plt.figure(figsize=(7, 6))

            sns.heatmap(
                cm,
                annot=True,
                fmt=".4f",
                cmap="coolwarm",
                cbar=True,
                square=True,
                linewidths=0.5,
                linecolor='gray',
                xticklabels=['MMN', 'OPMD', 'PL', 'BKG'],
                yticklabels=['MMN', 'OPMD', 'PL', 'BKG']
            )

            plt.xlabel("Predicted label", fontsize=14)
            plt.ylabel("True label", fontsize=14)
            plt.title(f"Confusion Matrix  - {label_name[model_name]}", fontsize=18)
            #plt.title(f"Confusion Matrix  - ", fontsize=20)

            plt.tight_layout()
            plt.savefig(f"./confusion_matrices/{model_file_name[:-4]}_confusion_matrix.png", dpi=300, bbox_inches="tight")
            plt.show()


if __name__ == "__main__":
    generate_confusion_matrix()
