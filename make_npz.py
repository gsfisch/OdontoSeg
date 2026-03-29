from PIL import Image
import numpy as np
import os


for file in os.listdir("./dataset/train/images"):
    img = Image.open(f'./dataset/train/images/{file}')

    arr = np.array(img)

    np.savez(f"Odonto_dataset_npz/{file}", image=arr)

'''

img = Image.open("image.jpg")

arr = np.array(img)

np.savez("image.npz", image=arr)
'''