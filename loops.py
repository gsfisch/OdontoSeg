import torch
import time
from tqdm import tqdm
from util.new_metrics import calculate_accuracy, calculate_mean_iou, calculate_precision_recall, calculate_dice_coefficient, calculate_weighted_accuracy, calculate_weighted_miou
from loss.main import criterion
from config import training_config

def compute_metrics_from_confusion_matrix(conf_matrix, eps=1e-7):
    """
    conf_matrix: (C, C)
    rows = ground truth
    cols = predictions
    """
    TP = torch.diag(conf_matrix)
    FP = conf_matrix.sum(dim=0) - TP
    FN = conf_matrix.sum(dim=1) - TP
    TN = conf_matrix.sum() - (TP + FP + FN)

    # Per-class metrics
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    iou = TP / (TP + FP + FN + eps)
    dice = (2 * TP) / (2 * TP + FP + FN + eps)

    # Means (include all classes, including background)
    mean_precision = precision.mean().item()
    mean_recall = recall.mean().item()
    mIoU = iou.mean().item()
    mean_dice = dice.mean().item()

    # Pixel accuracy
    accuracy = TP.sum() / conf_matrix.sum()

    # Frequency-weighted IoU
    freq = conf_matrix.sum(dim=1) / conf_matrix.sum()
    fwIoU = (freq * iou).sum()

    return {
        "accuracy": accuracy.item(),
        "precision": mean_precision,
        "recall": mean_recall,
        "mIoU": mIoU,
        "dice": mean_dice,
        "weighted_mIoU": fwIoU.item(),
    }

def compute_confusion_matrix(preds, targets, num_classes):
    """
    preds: (B, H, W) predicted labels
    targets: (B, H, W) ground truth labels
    """
    preds = preds.view(-1)
    targets = targets.view(-1)

    mask = (targets >= 0) & (targets < num_classes)
    preds = preds[mask]
    targets = targets[mask]

    conf_matrix = torch.bincount(
        num_classes * targets + preds,
        minlength=num_classes ** 2
    ).reshape(num_classes, num_classes)

    return conf_matrix


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

def train_loop(generator, optimizer, model, num_classes=4):
    model.train()
    
    # Initialize metrics
    running_metrics = {
        "loss": 0.0,
        "accuracy": 0.0,
        "mIoU": 0.0,
        "dice": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "weighted_accuracy": 0.0,
        "weighted_mIoU": 0.0
    }
    
    total_batches = len(generator)
    loss_function = training_config['loss_function']
    total_loss = 0.0
    total_samples = 0.0
    conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float64).cuda()
    
    
    loop = tqdm(enumerate(generator), total=total_batches, desc='Training')

    for batch_idx, (images, masks) in loop:
        torch.cuda.empty_cache()
        images = images.permute(0, 3, 1, 2).cuda()
        masks = masks.long().cuda()

        batch_size = images.size(0)
        total_samples += batch_size

        optimizer.zero_grad()
        outputs = model(images)

        loss = criterion(option=loss_function, outputs=outputs, masks=masks)
        loss.backward()
        optimizer.step()
        #time.sleep(training_config['delay_per_batch'])
        
        '''
        # Compute and accumulate metrics
        metrics = compute_metrics(outputs, masks)
        running_metrics["loss"] += loss.item()
        for key in running_metrics:
            if key != "loss":
                running_metrics[key] += metrics[key]
        '''
        total_loss += loss.item() * batch_size

        preds = torch.argmax(outputs, dim=1)
        conf_matrix += compute_confusion_matrix(preds, masks, num_classes)

        # Update tqdm description and postfix
        loop.set_description('train_batch {}/{}'.format(batch_idx + 1, total_batches))
        #loop.set_postfix({key: (value / (batch_idx + 1)) for key, value in running_metrics.items()})


    # Final metrics
    metrics = compute_metrics_from_confusion_matrix(conf_matrix)
    metrics["loss"] = total_loss / total_samples

    return metrics


        
    # Calculate average metrics
    #avg_metrics = {key: value / total_batches for key, value in running_metrics.items()}



def val_loop(generator, model, num_classes=4):
    model.eval()

    total_loss = 0.0
    total_samples = 0
    #generator_len = len(generator)

    conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float64).cuda()
    loss_function = training_config['loss_function']

    with torch.no_grad():

        loop = tqdm(generator, total=len(generator), desc='Validation')


        for images, masks in loop:
            images = images.permute(0, 3, 1, 2).cuda(non_blocking=True)
            masks = masks.long().cuda(non_blocking=True)

            batch_size = images.size(0)            
            
            total_samples += batch_size

            outputs = model(images)

            loss = criterion(
                option=loss_function,
                outputs=outputs,
                masks=masks
            )

            total_loss += loss.item() * batch_size

            preds = torch.argmax(outputs, dim=1)
            conf_matrix += compute_confusion_matrix(preds, masks, num_classes)

    # Final metrics
    metrics = compute_metrics_from_confusion_matrix(conf_matrix)
    metrics["loss"] = total_loss / total_samples

    return metrics

'''
def val_loop(generator, model):
    model.eval()
    
    running_metrics = {
        "loss": 0.0,
        "accuracy": 0.0,
        "mIoU": 0.0,
        "dice": 0.0,
        "precision": 0.0,
        "recall": 0.0,
        "weighted_accuracy": 0.0,
        "weighted_mIoU": 0.0
    }
    
    total_samples = 0
    loss_function = training_config['loss_function']

    conf_matrix = torch.zeros((num_classes, num_classes), dtype=torch.float64).cuda()

    with torch.no_grad():
    for images, masks in generator:
        images = images.permute(0, 3, 1, 2).cuda()
        masks = masks.long().cuda()

        outputs = model(images)
        preds = torch.argmax(outputs, dim=1)

        conf_matrix += compute_confusion_matrix(preds, masks, num_classes)

    with torch.no_grad():
        loop = tqdm(generator, total=len(generator), desc='Validation')

        for batch_idx, (images, masks) in enumerate(loop):
            # Move data
            images = images.permute(0, 3, 1, 2).cuda(non_blocking=True)
            masks = masks.long().cuda(non_blocking=True)

            batch_size = images.size(0)
            total_samples += batch_size

            # Forward pass
            outputs = model(images)

            # Loss (no clone!)
            loss = criterion(
                option=loss_function,
                outputs=outputs,
                masks=masks
            )

            # Compute metrics (must return per-batch averages)
            metrics = compute_metrics(outputs, masks)

            # Accumulate (weighted!)
            running_metrics["loss"] += loss.item() * batch_size

            for key in running_metrics:
                if key != "loss":
                    running_metrics[key] += metrics[key] * batch_size

            # Update progress bar with running averages
            loop.set_postfix({
                key: value / total_samples
                for key, value in running_metrics.items()
            })

    # Final averages
    avg_metrics = {
        key: value / total_samples
        for key, value in running_metrics.items()
    }

    return avg_metrics
'''

'''
def val_loop(generator, model):
    model.eval()
    
    # Initialize metrics
    running_metrics = {
        "loss": 0.0,
        "accuracy": 0.0,
        "mIoU": 0.0,
        "dice": 0.0,
        "precision": 0.0,
        "recall": 0.0,
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
            time.sleep(training_config['delay_per_batch'])

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


'''
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
        loop = tqdm(enumerate(generator),  total=total_batches, desc='Test')

        for batch_idx, (images, masks) in loop:
            masks = masks.long().cuda()
            images = images.permute(0, 3, 1, 2).cuda()
            outputs = model(images)
            loss = criterion(option=loss_function, outputs=outputs.clone(), masks=masks.clone())
            #time.sleep(training_config['delay_per_batch'])

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


'''
def train_loop(generator, optimizer, model):
    model.train()
    
    # Initialize metrics
    running_metrics = {
        "loss": 0.0,
        "accuracy": 0.0,
        "mIoU": 0.0,
        "dice": 0.0,
        "precision": 0.0,
        "recall": 0.0,
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
        time.sleep(training_config['delay_per_batch'])
        
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
'''
