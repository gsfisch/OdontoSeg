import os
import torch
from test import test_routine
from util.data import get_data_generators
from util.model import make_model
from loops import train_loop, val_loop
from optimizers.main import optimizer
import wandb
from datetime import datetime
from config import training_config, wandb_config, wandb_name, path_models
from util.scheduler import FlatplusAnneal, FlatplusAnnealTeste

def train():
    torch.cuda.empty_cache()
    
    # initialize model
    model = make_model(training_config['encoder'], training_config['architecture'], classes=training_config['classes']).cuda()
        
    # create experiment folder
    time_now = datetime.today().strftime('%Y%m%d-%H%M%S')
    folder_experiment = os.path.join(path_models, f"{training_config['experiment_name']}_{time_now}")
    os.makedirs(folder_experiment, exist_ok=True)
    
    # get data generators
    training_generator, valid_generator = get_data_generators()
    
    # initialize optimizer and scheduler
    opt = optimizer(
        option=training_config['optimizer'],
        model=model,
        lr=training_config['learning_rate'],
        weight_decay=training_config['weight_decay']
    )
    scheduler = FlatplusAnneal(opt, max_iter=300, step_size=0.7)
    
    # Initialize WandB
    wandb.init(
        project=wandb_name,
        name=training_config['experiment_name'],
        config=wandb_config
    )
    
    best_model_loss = float('inf')
    best_epoch = 0
    num_epochs = training_config['epochs']
    
    for epoch in range(num_epochs):
        # Train
        metrics_train = train_loop(training_generator, opt, model)

        # Validate
        metrics_val = val_loop(valid_generator, model)

        # Step scheduler
        scheduler.step()
        
        # Save model checkpoint
        model_path = os.path.join(folder_experiment, f'{training_config["experiment_name"]}-{epoch}.pt')
        torch.save(model.state_dict(), model_path)
        current_lr = opt.param_groups[0]['lr']

        # Update best model if needed
        if metrics_train['loss'] < best_model_loss:
            best_model_loss = metrics_train['loss']
            best_epoch = epoch
                
        # Print and log results
        print(f'\nEpoch {epoch + 1}/{num_epochs}, '
              f'train_loss: {metrics_train["loss"]:.3f}, val_loss: {metrics_val["loss"]:.3f}, '
              f'train_acc: {metrics_train["accuracy"]:.3f}, val_acc: {metrics_val["accuracy"]:.3f}, '
              f'train_mIoU: {metrics_train["mIoU"]:.3f}, val_mIoU: {metrics_val["mIoU"]:.3f}\n')
        
        metrics_wandb = logging_wandb(metrics_train, metrics_val)
        
        wandb.log({
            **metrics_wandb,
            'learning_rate': current_lr,
            'epoch': epoch
        })
        
    # Print best results
    print(f'Best train_loss: {best_model_loss:.3f}')
    print(f'Best epoch: {best_epoch}')
    
     # Run test routine
    test_routine(folder_experiment, best_epoch, wand_logged=True)
    
    # Finish WandB logging
    wandb.finish()

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
    