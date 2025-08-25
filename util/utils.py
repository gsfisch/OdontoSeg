import torch.nn as nn
import numpy as np
import torch
from skimage.io import imread, imsave
import pandas as pd
import pylab as plt
from torch import Tensor
from path_models import path

def extract_img(size, in_tensor):
    """
    Args:
        size(int) : size of cut
        in_tensor(tensor) : tensor to be cut
    """
    dim1, dim2 = in_tensor.size()[2:]
    in_tensor = in_tensor[:, :, int((dim1-size)/2):int((dim1+size)/2),
                          int((dim2-size)/2):int((dim2+size)/2)]
    return in_tensor

def extract_img_new(x, oldConv):
    current_layer_dim = x.shape[2]
    old_layer_dim = oldConv.shape[2]


    lower = int((old_layer_dim - current_layer_dim) / 2)
    upper = int(old_layer_dim - lower)
    conv4_out_modified = oldConv[:, :, lower:upper, lower:upper]

    x = torch.cat([x, conv4_out_modified], dim=1)

    return x

def input_filled_mirroring(x, e = 62):      # fill missing data by mirroring the input image
    '''input size 636 --> output size 512'''
    # w, h = x.shape
    w, h = np.shape(x)[0], np.shape(x)[1]
    #e = 62  # extra width on 1 edge
    y = np.zeros((h + e * 2, w + e * 2, 3))    
   
    y[e:h + e, e:w + e, :] = x
    y[e:e + h, 0:e, :] = np.flip(y[e:e + h, e:2 * e, :], 1)  # flip vertically
    y[e:e + h, e + w:2 * e + w, :] = np.flip(y[e:e + h, w:e + w, :], 1)  # flip vertically
    y[0:e, 0:2 * e + w, :] = np.flip(y[e:2 * e, 0:2 * e + w, :], 0)  # flip horizontally
    y[e + h:2 * e + h, 0:2 * e + w, :] = np.flip(y[h:e + h, 0:2 * e + w, :], 0)  # flip horizontally
    
    return y

def input_filled_mirroring_grayscale(x, e = 62):      # fill missing data by mirroring the input image
    '''input size 636 --> output size 512'''
    # w, h = x.shape
    w, h = np.shape(x)[0], np.shape(x)[1]
    #e = 62  # extra width on 1 edge
    y = np.zeros((h + e * 2, w + e * 2))    
   
    y[e:h + e, e:w + e] = x
    y[e:e + h, 0:e] = np.flip(y[e:e + h, e:2 * e], 1)  # flip vertically
    y[e:e + h, e + w:2 * e + w] = np.flip(y[e:e + h, w:e + w], 1)  # flip vertically
    y[0:e, 0:2 * e + w] = np.flip(y[e:2 * e, 0:2 * e + w], 0)  # flip horizontally
    y[e + h:2 * e + h, 0:2 * e + w] = np.flip(y[h:e + h, 0:2 * e + w], 0)  # flip horizontally
    
    return y

def save_df(df, epoch, train_loss, val_loss, train_acc, val_acc):
    df.at[epoch, 'EPOCH'] = epoch
    df.at[epoch, 'VAL_LOSS'] = val_loss
    df.at[epoch, 'TRAIN_LOSS'] = train_loss
    df.at[epoch, 'TRAIN_ACC'] = train_acc
    df.at[epoch, 'VAL_ACC'] = val_acc

    return df

def get_usage():
    for obj in gc.get_objects():
        try:
            if torch.is_tensor(obj) or (hasattr(obj, 'data') and torch.is_tensor(obj.data)):
                print(type(obj), obj.size())
        except: pass


def plot_loss(df, experiment_name):
  plt.plot(df['EPOCH'].values.tolist(), df['VAL_LOSS'].values.tolist(), df['TRAIN_LOSS'].values.tolist())
  plt.xlabel('iteration')
  plt.ylabel('loss')
  plt.legend(['VAL_LOSS', 'TRAIN_LOSS'])

  plt.savefig(path + '/graphs/' + experiment_name +'.png')

# center crop image
def center_crop(image, final_size): 
    h, w = image.shape

    padding = int ( (h - 512) / 2 )  
    new_img = np.zeros((final_size, final_size))
    new_img = image[padding:final_size+padding, padding:final_size +padding]

    return new_img

def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


def flatten(tensor):
    C = tensor.size(1)
    # new axis order
    axis_order = (1, 0) + tuple(range(2, tensor.dim()))
    # Transpose: (N, C, D, H, W) -> (C, N, D, H, W)
    transposed = tensor.permute(axis_order)
    # Flatten: (C, N, D, H, W) -> (C, N * D * H * W)
    return transposed.contiguous().view(C, -1)

