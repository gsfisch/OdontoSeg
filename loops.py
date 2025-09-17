import torch
import time
from tqdm import tqdm
from util.new_metrics import calculate_accuracy, calculate_mean_iou, calculate_precision_recall, calculate_dice_coefficient, calculate_weighted_accuracy, calculate_weighted_miou
from loss.main import criterion
from config import training_config


def compute_metrics(outputs, masks):
    """
    Compute all metrics and return a dictionary with their values.
    """
    precision, recall = calculate_precision_recall(outputs, masks)
    return {
        "accuracy": calculate_accuracy(outputs, masks).item(),
        "mIoU": calculate_mean_iou(outputs, masks).item(),
        "precision": precision.item(),
        "recall": recall.item(),
        "dice": calculate_dice_coefficient(outputs, masks).item(),
        "weighted_accuracy": calculate_weighted_accuracy(outputs, masks).item(),
        "weighted_mIoU": calculate_weighted_miou(outputs, masks).item()
    }

def train_loop(generator, optimizer, model):
    model.train()
    
    # Initialize metrics
    running_metrics = {
        "loss": 0.0,
        "accuracy": 0.0,
        "mIoU": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "dice": 0.0,
        "weighted_accuracy": 0.0,
        "weighted_mIoU": 0.0
    }
    
    total_batches = len(generator)
    loss_function = training_config['loss_function']
    
    loop = tqdm(enumerate(generator), total=total_batches, desc='Training')

    for batch_idx, (images, masks) in loop:
        torch.cuda.empty_cache()
        images = images.permute(0, 3, 1, 2).cuda()
        masks = masks.long().cuda()
        
        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(option=loss_function, outputs=outputs, masks=masks)
        loss.backward()
        optimizer.step()
        time.sleep(2)
        
        # Compute and accumulate metrics
        metrics = compute_metrics(outputs, masks)
        running_metrics["loss"] += loss.item()
        for key in running_metrics:
            if key != "loss":
                running_metrics[key] += metrics[key]

        # Update tqdm description and postfix
        loop.set_description('train_batch {}/{}'.format(batch_idx + 1, total_batches))
        loop.set_postfix({key: (value / (batch_idx + 1)) for key, value in running_metrics.items()})
        
    # Calculate average metrics
    avg_metrics = {key: value / total_batches for key, value in running_metrics.items()}

    return avg_metrics

def val_loop(generator, model):
    model.eval()
    
    # Initialize metrics
    running_metrics = {
        "loss": 0.0,
        "accuracy": 0.0,
        "mIoU": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "dice": 0.0,
        "weighted_accuracy": 0.0,
        "weighted_mIoU": 0.0
    }
    
    loss_function = training_config['loss_function']
    total_batches = len(generator)
    
    with torch.no_grad():
        loop = tqdm(enumerate(generator),  total=total_batches, desc='Validation')

        for batch_idx, (images, masks) in loop:
            masks = masks.long().cuda()
            images = images.permute(0, 3, 1, 2).cuda()
            outputs = model(images)
            loss = criterion(option=loss_function, outputs=outputs.clone(), masks=masks.clone())

            # Compute and accumulate metrics
            metrics = compute_metrics(outputs, masks)
            running_metrics["loss"] += loss.item()
            for key in running_metrics:
                if key != "loss":
                    running_metrics[key] += metrics[key]

             # Update tqdm description and postfix
            loop.set_description('val_batch {}/{}'.format(batch_idx + 1, total_batches))
            loop.set_postfix({key: (value / (batch_idx + 1)) for key, value in running_metrics.items()})
            
    # Calculate average metrics
    avg_metrics = {key: value / total_batches for key, value in running_metrics.items()}
    return avg_metrics

def test_loop(generator, model):
    model.eval()
    
    # Initialize metrics
    running_metrics = {
        "loss": 0.0,
        "accuracy": 0.0,
        "mIoU": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "dice": 0.0,
        "weighted_accuracy": 0.0,
        "weighted_mIoU": 0.0
    }
    
    loss_function = training_config['loss_function']
    total_batches = len(generator)
    
    with torch.no_grad():
        loop = tqdm(enumerate(generator),  total=total_batches, desc='Validation')

        for batch_idx, (images, masks) in loop:
            masks = masks.long().cuda()
            images = images.permute(0, 3, 1, 2).cuda()
            outputs = model(images)
            loss = criterion(option=loss_function, outputs=outputs.clone(), masks=masks.clone())

            # Compute and accumulate metrics
            metrics = compute_metrics(outputs, masks)
            running_metrics["loss"] += loss.item()
            for key in running_metrics:
                if key != "loss":
                    running_metrics[key] += metrics[key]

             # Update tqdm description and postfix
            loop.set_description('val_batch {}/{}'.format(batch_idx + 1, total_batches))
            loop.set_postfix({key: (value / (batch_idx + 1)) for key, value in running_metrics.items()})
            
    # Calculate average metrics
    avg_metrics = {key: value / total_batches for key, value in running_metrics.items()}
    return avg_metrics