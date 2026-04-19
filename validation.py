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


def validate():
    torch.cuda.empty_cache()

    model_directory_path = validation_config['model_directory_path']
    configs_file_name = validation_config['configs_file_name']
    device = torch.device('cuda')
    model_file_name = validation_config['model_file_name']


    # Read training configurations
    training_config = {}
    with open(os.path.join(model_directory_path, configs_file_name), 'r') as configs_file:
        training_config = ast.literal_eval(configs_file.read())


    # Initialize and load model
    if training_config['library'] == 'smp':
        model = make_model(training_config['encoder'], training_config['architecture'],
                            training_config['classes'], library='smp',
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
    training_generator, valid_generator, test_generator = get_data_generators(training_config['batch_size'])

    with torch.no_grad():

        # Validate
        metrics_val = val_loop(test_generator, model)


        # Print and log results
        print(  f'val_loss: {metrics_val["loss"]:.4f}\n' +
                f'val_acc: {metrics_val["accuracy"]:.4f}\n' +
                f'val_mIoU: {metrics_val["mIoU"]:.4f}\n' +
                f'val_dice: {metrics_val["dice"]:.4f}\n' +
                f'val_precision: {metrics_val["precision"]:.4f}\n' +
                f'val_recall: {metrics_val["recall"]:.4f}\n'
            )


if __name__ == "__main__":
    validate()
