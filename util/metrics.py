import torch
from skimage.io import imread, imsave
import timeit
from sklearn.metrics import jaccard_score
import numpy as np
"""
  Calculate the pixel accuracy

  input (Tensor): (N, C, H, W) - float / network outputs
  target (Tensor): (N, H, W) - long / masks ground truth
"""
def calculate_accuracy(input, target):
  _, predicted = torch.max(input, 1)

  predicted_1d = predicted.view(-1)  
  targs = target.view(-1) 
  
  return (predicted_1d==targs).float().mean()

"""
  Calculate the pixel accuracy for a binary class problem

  input (Tensor): (N, H, W) - float / network outputs
  target (Tensor): (N, H, W) - long / masks ground truth
"""

def calculate_accuracy_binary(input, target):
  input = outputs.view(-1)  
  targs = local_labels.view(-1) 
  
  input[input >= 0.5] = 1
  input[input < 0.5] = 0
  
  return (input==targs).float().mean()

def calculate_mean_iou_3(input: torch.Tensor, target: torch.Tensor):
    _, predicted = torch.max(input.clone(), 1)
    one_d_target = target.clone().view(2, -1)
    one_d_predicted = predicted.view(2, -1)

    SMOOTH = 1e-6
    # print (one_d_target[0] == 1)
    # print ((one_d_target[0] == 1).float().sum())
    # print ((one_d_target[0] == 2).float().sum())
    # print ((one_d_predicted[0] == 1).float().sum())
    # print ((one_d_predicted[0] == 2).float().sum())

    
    intersection = (predicted & target).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (predicted | target).float().sum((1, 2))         # Will be zero if both are 0
    print ("\n")
    print ("init")
    print ("micro 1")
    a = jaccard_score(one_d_target[0].tolist(), one_d_predicted[0].tolist(), average=None)
    a2 = jaccard_score(one_d_target[1].tolist(), one_d_predicted[1].tolist(), average=None)
    print (a, a2)
    print ((np.mean(a) + np.mean(a2)) / 2.0 )
    print (calculate_mean_iou_3(input.clone(), target.clone()).mean())
    iou = (intersection + SMOOTH) / (union + SMOOTH)
    print (iou.mean())
    print ("end")
    # # print (jaccard_score(one_d_target.tolist(), one_d_predicted.tolist(), average='samples'))
    # print (iou.mean())

    return (iou.mean())

def calculate_mean_iou(input, target, eps=1e-7):
    """
      Calculate the mIOU 

      input (Tensor): (N, C, H, W) - float / network outputs
      target (Tensor): (N, H, W) - long / masks ground truth
    """

    batch_size = input.size(0)
    _, predicted = torch.max(input, 1)
    one_d_predicted = predicted.view(batch_size, -1)
    one_d_true = target.view(batch_size, -1)
    num_classes = input.size(1)
    iou_total = 0.0
    
    for i in range(batch_size):
      num_classes_batch = 0.0
      iou = 0.0
      for classe in range(num_classes):
        predicted_equal_class = (one_d_predicted[i] == classe)
        true_equal_predicted = (one_d_true[i] == one_d_predicted[i])
        true_equal_class = (one_d_true[i] == classe)
        predicted_equal_class = (one_d_predicted[i] == classe)


        tp = ((predicted_equal_class * true_equal_predicted)).float().sum()
        fp = (predicted_equal_class * ~true_equal_predicted).float().sum()
        fn = (true_equal_class * ~true_equal_predicted).float().sum()
        num_pixel = (true_equal_class).float().sum()
        num_pixel_predicted = (predicted_equal_class).float().sum()

        if (num_pixel != 0):
          num_classes_batch += 1
          iou += (tp / (tp + fp + fn + eps))
        elif (num_pixel == 0 and num_pixel_predicted != 0):
          num_classes_batch += 1
      if num_classes_batch != 0:
        iou_total += (iou / num_classes_batch) 

    return (iou_total / batch_size)

def calculate_accuracy_eval(input, target):
    predicted_1d = input.view(-1).cuda()  
    targs = target.view(-1).cuda()  
    
    return (predicted_1d==targs).float().mean()

def calculate_miou_eval(input, target, miou, number, eps=1e-7):
    """
      Calculate the mIOU 

      input (Tensor): (H, W) - float / network outputs
      target (Tensor): (H, W) - long / masks ground truth
    """
    # start = time.time()
    one_d_input = input.view(1, -1).cuda()
    one_d_true = target.view(1, -1).cuda()
    iou = torch.zeros(3).float().cuda()
    num_classes = 3
    num_classes_batch = 0.0

    for classe in range(num_classes):
      tp = ((one_d_input == classe) * (one_d_true == one_d_input)).float().sum()
      fp = ((one_d_input == classe) * (one_d_true != one_d_input)).float().sum()
      fn = ((one_d_true == classe) * (one_d_true != one_d_input)).float().sum()
      num_pixel = ((one_d_true == classe)).float().sum() 
      predicted_equal_class = (one_d_input == classe).float().sum()

      if (num_pixel != 0 or predicted_equal_class != 0):
        num_classes_batch += 1
        iou[classe] = (tp / (tp + fp + fn + eps))
        miou[classe] = (tp / (tp + fp + fn + eps))
        # miou[classe] = miou[classe] + (tp / (tp + fp + fn + eps))
        number[classe] = number[classe] + 1

    iou_value = (iou.sum() / num_classes_batch).item()

    return (iou_value, miou, number, iou)
#oldmetrics
# def calculate_accuracy(gtTorch, finalImgTorch):
#     red = (gtTorch[:,:,0].view(-1)==finalImgTorch[:,:,0].view(-1))
#     green = (gtTorch[:,:,1].view(-1)==finalImgTorch[:,:,1].view(-1))
#     blue = (gtTorch[:,:,2].view(-1)==finalImgTorch[:,:,2].view(-1))

#     rgb = (red*green*blue)

#     acc = rgb.float().mean()

#     return acc

# def mask_softmax_toclass(final_softmax):
#   final_img = final_softmax.detach().cpu().numpy()
#   check_image = final_img.copy()
#   result = np.zeros((512, 512, 3))
  
#   for i in range(512):
#     for j in range(512):
#       class_0 = check_image[0, i, j]
#       class_1 = check_image[1, i, j]
#       class_2 = check_image[2, i, j]
#       final_class = 0

#       if (class_0 > class_1 and class_0 > class_2):
#         final_class = 0
#         final_value = class_0
#       elif ((class_1 > class_0 and class_1 > class_2)):
#         final_class = 1
#         final_value = class_1
#       else:
#         final_class = 2
#         final_value = class_2


#       if (final_value >= 0.9):
#         result[i, j, :] = class_predicted[final_class]
#       else:
#         result[i, j, :] = class_predicted[0]


#   return result