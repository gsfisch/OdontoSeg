import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
from util.data import get_data_generators
from util.model import make_model
from loops import val_loop, test_loop
from optimizers.main import optimizer
import wandb
from datetime import datetime
from validation_config import validation_config
#from util.scheduler import FlatplusAnneal, FlatplusAnnealTeste
#from model_src import SegmentationModel
from torchinfo import summary
import torchseg
import ast
import numpy as np



models_to_validate = [
    #'efficientnet-b6_U-Net',
    #'efficientnet-b6_Linknet', 'efficientnet-b6_FPN',
    #'resnet34_U-Net', 'SegFormer_mit_b0', 'SegFormer_mit_b1', 'SegFormer_mit_b2', 'SegFormer_mit_b3',
    #'SegFormer_mit_b4', 'SegFormer_mit_b5', 'vgg16_FPN', 'vgg16_Linknet', 'vgg16_U-Net',
    #'vit_large_patch16_224_FPN', 'vit_large_patch16_224_MAnet', 
    'vit_large_patch16_224_Linknet', 
    'vit_large_patch16_224_U-Net',
    #'vit_large_patch16_224_U-Net++', 
]



def validate():
    torch.cuda.empty_cache()
    device = torch.device('cuda')


    for model_name in models_to_validate:
        model_directory_path = './models/' + model_name
        configs_file_name = 'config.txt'
        logs_file_name = f'logs_{model_name}.txt'
        
        # Read training configurations
        training_config = {}
        with open(os.path.join(model_directory_path, configs_file_name), 'r') as configs_file:
            training_config = ast.literal_eval(configs_file.read())

        with open(os.path.join(model_directory_path, logs_file_name), 'w') as logs_file:
        
            best_alternative = 0
            best_dice = float('-inf')

            logs_file.write('Validation\n\n')

            val_metrics = {
                'precision': [],
                'recall': [],
                'dice': [],
                'miou': []
            }

            for i in range(1, 6):
                model_file_name = model_name + f'_alternative_{i}.pth'


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
                    metrics_val = val_loop(valid_generator, model)


                    # Print and log results
                    print(  f'val_loss: {metrics_val["loss"]:.4f}\n' +
                            f'val_acc: {metrics_val["accuracy"]:.4f}\n' +
                            f'val_mIoU: {metrics_val["mIoU"]:.4f}\n' +
                            f'val_dice: {metrics_val["dice"]:.4f}\n' +
                            f'val_precision: {metrics_val["precision"]:.4f}\n' +
                            f'val_recall: {metrics_val["recall"]:.4f}\n'
                        )

                    # Save current metrics
                    val_metrics['precision'].append(metrics_val['precision'])
                    val_metrics['recall'].append(metrics_val['recall'])
                    val_metrics['dice'].append(metrics_val['dice'])
                    val_metrics['miou'].append(metrics_val['mIoU'])

                    logs_file.write(
                            f'Alternative {i}\n' +
                            f'val_loss: {metrics_val["loss"]:.4f}\n' +
                            f'val_acc: {metrics_val["accuracy"]:.4f}\n' +
                            f'val_mIoU: {metrics_val["mIoU"]:.4f}\n' +
                            f'val_dice: {metrics_val["dice"]:.4f}\n' +
                            f'val_precision: {metrics_val["precision"]:.4f}\n' +
                            f'val_recall: {metrics_val["recall"]:.4f}\n'
                    )


                    if metrics_val["dice"] > best_dice:
                        best_alternative = i
                        best_dice = metrics_val["dice"]
                        print(f'New best: {best_alternative=}')

            

            training_generator, valid_generator, test_generator = get_data_generators(training_config['batch_size'], dataset_path=validation_config['dataset_path'])

            # Initialize and load best model
            if training_config['library'] == 'smp':
                best_model = make_model(training_config['encoder'], training_config['architecture'],
                                        training_config['classes'], library='smp',
                                        decoder_channels=training_config['decoder_channels'], encoder_depth=training_config['encoder_depth'],
                                        freeze_encoder=training_config['freeze_encoder'], need_wrapper=training_config['need_wrapper']).cuda()

            else:    
                best_model = make_model(training_config['encoder'], training_config['architecture'], 
                                classes=training_config['classes'], library=training_config['library'],
                                decoder_channels=training_config['decoder_channels'], encoder_depth=training_config['encoder_depth'],
                                encoder_params=training_config['encoder_params'], head_upsampling=training_config['head_upsampling'],
                                freeze_encoder=training_config['freeze_encoder'], need_wrapper=training_config['need_wrapper']).cuda()


            model_file_name = model_name + f'_alternative_{best_alternative}.pth'

            best_model.load_state_dict(torch.load(os.path.join(model_directory_path, model_file_name), weights_only="True"))

            with torch.no_grad():
                # Test
                metrics_val = val_loop(test_generator, best_model)

                logs_file.write(
                            f'Test\n\n' +
                            f'Alternative {best_alternative}\n' +
                            f'val_loss: {metrics_val["loss"]:.4f}\n' +
                            f'val_acc: {metrics_val["accuracy"]:.4f}\n' +
                            f'val_mIoU: {metrics_val["mIoU"]:.4f}\n' +
                            f'val_dice: {metrics_val["dice"]:.4f}\n' +
                            f'val_precision: {metrics_val["precision"]:.4f}\n' +
                            f'val_recall: {metrics_val["recall"]:.4f}\n' +

                            f'Validation Mean and SD\n\n' +
                            f'Precision: ' + str(round(np.mean(val_metrics['precision']), 4)) + ' +- ' + str(round(np.std(val_metrics['precision']), 4)) + '\n' +
                            f'Recall: ' + str(round(np.mean(val_metrics['recall']), 4)) + ' +- ' + str(round(np.std(val_metrics['recall']), 4)) + '\n' +
                            f'Dice: ' + str(round(np.mean(val_metrics['dice']), 4)) + ' +- ' + str(round(np.std(val_metrics['dice']), 4)) + '\n' +
                            f'mIoU: ' + str(round(np.mean(val_metrics['miou']), 4)) + ' +- ' + str(round(np.std(val_metrics['miou']), 4)) + '\n'
                    )

                



if __name__ == "__main__":
    validate()
