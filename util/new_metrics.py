import torch
from config import training_config

def calculate_mean_iou(input, target, eps=1e-7):
    """
      Calculate the mIOU 

      input (Tensor): (N, C, H, W) - float / network outputs
      target (Tensor): (N, H, W) - long / masks ground truth
    """

    batch_size = input.size(0)
    num_classes = input.size(1)
    
    _, predicted = torch.max(input, 1)
    one_d_predicted = predicted.view(batch_size, -1)
    one_d_true = target.view(batch_size, -1)
    
    iou_total = 0.0
    
    for i in range(batch_size):
      num_classes_batch = 0.0
      iou_sum = 0.0
      
      for cls in range(num_classes):
        predicted_equal_class = (one_d_predicted[i] == cls)
        true_equal_class = (one_d_true[i] == cls)
        
        tp = (predicted_equal_class * true_equal_class).float().sum()
        fp = (predicted_equal_class * ~true_equal_class).float().sum()
        fn = (true_equal_class * ~predicted_equal_class).float().sum()
        
        if tp + fp + fn > 0:
          iou_sum += tp / (tp + fp + fn + eps)
          num_classes_batch += 1
          
      if num_classes_batch > 0:
        iou_total += (iou_sum / num_classes_batch) 

    return (iou_total / batch_size)
  
def calculate_accuracy(input, target):
    _, predicted = torch.max(input, 1)  # Obtém as predições de classe
    
    predicted_1d = predicted.view(-1)  
    targs = target.view(-1) 
    
    return (predicted_1d == targs).float().mean()

def calculate_precision_recall(input, target, eps=1e-7):
    num_classes = input.size(1)
    precision_total = 0.0
    recall_total = 0.0
    for cls in range(num_classes):
        predicted_class = (input.argmax(dim=1) == cls).float()
        true_class = (target == cls).float()
        
        intersection = (predicted_class * true_class).sum()
        
        precision = intersection / (predicted_class.sum() + eps)
        recall = intersection / (true_class.sum() + eps)
        
        precision_total += precision
        recall_total += recall
    return precision_total / num_classes, recall_total / num_classes
  

def calculate_dice_coefficient(input, target, eps=1e-7):
    num_classes = input.size(1)
    dice_total = 0.0
    for cls in range(num_classes):
        predicted_class = (input.argmax(dim=1) == cls).float()
        true_class = (target == cls).float()
        intersection = (predicted_class * true_class).sum()
        union = predicted_class.sum() + true_class.sum()
        dice_total += (2. * intersection + eps) / (union + eps)
    return dice_total / num_classes

def calculate_weighted_accuracy(input, target):
    # Convert input and target to 1D tensors
    device = input.device
    predicted_1d = input.argmax(dim=1).view(-1)
    target_1d = target.view(-1)
    
    num_classes = input.size(1)
    
    # Ensure class_weights is a tensor
    class_weights = torch.tensor(training_config['class_weigths'], device=device)

    # Initialize total weighted accuracy and weight sum
    total_weighted_accuracy = torch.tensor(0.0, device=device)
    total_weight = torch.tensor(0.0, device=device)

    # Compute weighted accuracy for each class
    for cls in range(num_classes):
        predicted_class = (predicted_1d == cls)
        true_class = (target_1d == cls)
        correct_predictions = (predicted_class & true_class).float().sum()
        total_class_pixels = true_class.float().sum()

        if total_class_pixels > 0:
            accuracy = correct_predictions / total_class_pixels
        else:
            accuracy = torch.tensor(0.0, device=device)
        
        # Apply class weight
        weight =  torch.tensor(class_weights[cls])
        total_weighted_accuracy += weight * accuracy
        total_weight += weight

    # Normalize by the total weight
    if total_weight > 0:
        return total_weighted_accuracy / total_weight
    else:
        return  torch.tensor(0.0, device=device)
  
def calculate_weighted_miou(input, target, eps=1e-7):
    predicted_1d = input.argmax(dim=1).view(-1)
    target_1d = target.view(-1)
    num_classes = input.size(1)
    class_weights = training_config['class_weigths']
    
    iou_scores = []
    for cls in range(num_classes):
        predicted_class = (predicted_1d == cls)
        true_class = (target_1d == cls)
        
        intersection = (predicted_class * true_class).float().sum()
        union = (predicted_class + true_class).float().sum() - intersection
        iou = (intersection + eps) / (union + eps)
        
        iou_scores.append(class_weights[cls] * iou)

    return sum(iou_scores) / sum(class_weights)