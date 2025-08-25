import numpy as np
from skimage.io import imread, imsave
import glob
import tqdm

carcioma_paths = sorted(glob.glob('../data_resized/masks/carcinomas/*.png'))
leucoplasia_paths = sorted(glob.glob('../data_resized/masks/leucoplasias/*.png'))

total_paths = carcioma_paths + leucoplasia_paths


def checkPixels_gt(arr_test, fileName):
    h, w = 512, 512
    normal, green, red, error = 0, 0, 0, 0

    print (fileName)
    for i in range(h):
      for j in range(w):
        if (arr_test[i, j, 0] == 0 and arr_test[i, j, 1] == 0 and arr_test[i, j, 2] == 255):
          normal = normal + 1
        elif (arr_test[i, j, 0] == 0 and arr_test[i, j, 1] == 255 and arr_test[i, j, 2] == 0):  
          green = green + 1
        elif (arr_test[i, j, 0] == 255 and arr_test[i, j, 1] == 0 and arr_test[i, j, 2] == 0):
          red = red + 1
        else:
          error = error + 1

    if (error != 0):
      print ("ERROR")

    return normal, green, red


def mask_to_class(mask, fileName):
    ground_truth = mask.copy()
    total = []
    total_normal = 0
    total_green = 0 
    total_red = 0

    
    return checkPixels_gt(ground_truth, fileName)

  
total_normal = 0
total_green = 0 
total_red = 0

for i in tqdm.tqdm(range(len(total_paths))):
  fileName = total_paths[i]

  normal, green, red = mask_to_class(imread(total_paths[i]), fileName)
  total_normal += normal
  total_green += green
  total_red += red

freq_n = float(total_normal)/ (len(total_paths)*512*512)
freq_r = float(total_red)/ (len(carcioma_paths)*512*512)
freq_g = float(total_green)/ (len(leucoplasia_paths)*512*512)

print ('normal: ' + str(total_normal))
print ('green: ' + str(total_green))
print ('red: ' + str(total_red))

print ('freq normal: ' + str(freq_n))
print ('freq green: ' + str(freq_g))
print ('freq red: ' + str(freq_r))

weight_n = freq_r / freq_n
weight_r = freq_r / freq