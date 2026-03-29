import cv2
import numpy as np
import os

# Allowed colors (BGR because OpenCV uses BGR)
palette = np.array([
    [0, 0, 255],      # Red
    [0, 255, 255],    # Yellow
    [0, 255, 0],      # Green
    [255, 0, 0]       # Blue
], dtype=np.uint8)


original_dataset_path = 'new_dataset_sri_lanka/'
new_dataset_path = 'new_new_dataset_sri_lanka/'

sets = ['train', 'validation', 'test']

for current_set in sets:
    for img_file in os.listdir(original_dataset_path + current_set + '/masks'):
        print(img_file)
        
        img = cv2.imread(original_dataset_path + current_set + '/masks/' + img_file)
        h, w, _ = img.shape

        # Reshape for easier computation
        pixels = img.reshape((-1, 3))

        # Compute distances
        distances = np.linalg.norm(pixels[:, None] - palette[None], axis=2)

        # Replace each pixel with closest palette color
        new_pixels = palette[np.argmin(distances, axis=1)]

        # Reshape back
        result = new_pixels.reshape((h, w, 3))

        cv2.imwrite(new_dataset_path + current_set + '/masks/' + img_file, result)