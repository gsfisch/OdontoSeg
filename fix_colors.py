import os
from PIL  import Image
import matplotlib.pyplot as plt
import numpy as np


training_masks_dir = 'dataset_sri_lanka/train/masks'
validation_masks_dir = 'dataset_sri_lanka/validation/masks'
test_masks_dir = 'dataset_sri_lanka/test/masks'

# Test
for mask_name in os.listdir('dataset_sri_lanka/test/masks'):
    img = np.array(Image.open(os.path.join(test_masks_dir, mask_name)))
    height, width, _ = img.shape

    plt.imshow(img)
    plt.show()
    
    for i in range(height):
        for j in range(width):
            if img[i, j, 2] != 254:
                print(img[i, j, :])
            if img[i, j, 0] == 0 and img[i, j, 1] == 255 and img[i, j, 2] == 0: # Green
                img[i, j, 0] == 255
                img[i, j, 1] == 255  # Turn into yellow
                img[i, j, 2] == 0

            elif img[i, j, 0] == 255 and img[i, j, 1] == 255 and img[i, j, 2] == 0: # Yellow
                img[i, j, 0] == 0 
                img[i, j, 1] == 255 # Turn into green
                img[i, j, 2] == 0

    plt.imshow(img)
    plt.show()
