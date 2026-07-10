import os
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
import torch
#from test import test_routine
from util.data import get_data_generators
from util.model import make_model
from loops import train_loop, val_loop
from optimizers.main import optimizer
import wandb
from datetime import datetime
from config import training_config, wandb_config, wandb_name, path_models
from util.scheduler import FlatplusAnneal, FlatplusAnnealTeste
from torchinfo import summary
import torchseg


def train():
    for i in range(2, 6):
        torch.cuda.empty_cache()
        torch.manual_seed(i)
        experiment_name = training_config['experiment_name']
        num_epochs = training_config['epochs']
        epoch_to_unfreeze_encoder = 300
        #device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


        # initialize model
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
        

        summary(model, input_size=(training_config['batch_size'], 3, 512, 512))

        
        # get data generators
        training_generator, valid_generator, test_generator = get_data_generators()
        
        # initialize optimizer and scheduler
        opt = optimizer(
            option=training_config['optimizer'],
            model=model,
            lr=training_config['learning_rate'],
            weight_decay=training_config['weight_decay']
        )
        #scheduler = FlatplusAnneal(opt, max_iter=1000, step_size=0.7)
        scheduler = FlatplusAnneal(opt, max_iter=training_config['epochs'], step_size=training_config['scheduler_step_size'])
        
        # Initialize WandB
        wandb.init(
            project=wandb_name,
            name=f"{training_config['experiment_name']}" + "_alternative_{i}",
            config=wandb_config
        )
        
        best_model_loss = float('inf')
        best_model_dice = float('-inf')

        # create experiment folder
        folder_experiment = os.path.join(path_models, f"{experiment_name}_alternative_{i}")
        print(f"Creating folder: {folder_experiment}")
        os.makedirs(folder_experiment, exist_ok=False)

        
        for epoch in range(num_epochs):
            # Unfreeze encoder after warmup
            '''
            if epoch >= epoch_to_unfreeze_encoder - 1:
                # If no wrapper was used
                if hasattr(model, "encoder"):
                    for param in model.encoder.parameters():
                        param.requires_grad = True

                # If wrapper was used 
                elif hasattr(model, "model") and hasattr(model.model, "encoder"):
                    for param in model.model.encoder.parameters():
                        param.requires_grad = True
            '''

            # Train
            metrics_train = train_loop(training_generator, opt, model)

            # Validate
            metrics_val = val_loop(valid_generator, model)

            # Step scheduler
            scheduler.step()

            current_lr = opt.param_groups[0]['lr']

            # Print and log results
            print(f'\nEpoch {epoch}/{num_epochs - 1}, '
                f'train_loss: {metrics_train["loss"]:.4f}, val_loss: {metrics_val["loss"]:.4f}, '
                f'train_acc: {metrics_train["accuracy"]:.4f}, val_acc: {metrics_val["accuracy"]:.4f}, '
                f'train_mIoU: {metrics_train["mIoU"]:.4f}, val_mIoU: {metrics_val["mIoU"]:.4f}\n ',
                f'train_dice: {metrics_train["dice"]:.4f}, val_dice: {metrics_val["dice"]:.4f}\n',
                f'train_precision: {metrics_train["precision"]:.4f}, val_precision: {metrics_val["precision"]:.4f}\n',
                f'train_recall: {metrics_train["recall"]:.4f}, val_recall: {metrics_val["recall"]:.4f}\n'
                )
                
            # Check best checkpoint based on Validation Dice
            if metrics_val['dice'] > best_model_dice:
                best_model_dice = metrics_val['dice']
                
                model_path = os.path.join(folder_experiment, f'{experiment_name}_alternative_{i}.pth')
                torch.save(model.state_dict(), model_path)

                print(f"Saving Checkpoint at epoch: {epoch}")
            
            metrics_wandb = logging_wandb(metrics_train, metrics_val)
            
            wandb.log({
                **metrics_wandb,
                'learning_rate': current_lr,
                'epoch': epoch
            })

        wandb.finish()

    return


def logging_wandb(metrics_train, metrics_validation):
    return {
        "train/loss": metrics_train['loss'],
        "train/accuracy": metrics_train['accuracy'],
        "train/mIoU": metrics_train['mIoU'],
        "train/precision": metrics_train['precision'],
        "train/recall":metrics_train['recall'],
        "train/dice": metrics_train['dice'],
        "eval/loss": metrics_validation['loss'],
        "eval/accuracy": metrics_validation['accuracy'],
        "eval/mIoU": metrics_validation['mIoU'],
        "eval/precision": metrics_validation['precision'],
        "eval/recall": metrics_validation['recall'],
        "eval/dice": metrics_validation['dice'],
    }

if __name__ == "__main__":
    train()
