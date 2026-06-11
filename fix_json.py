# This script was used to add the missing 'Height and 'Width' attributes in the JSON file
import json
from PIL import Image
import numpy as np


with open("new_dataset/Annotation.json") as f:
    data = json.load(f)

    print("JSON valid. Keys:", data.keys())

    i = 0
    for img in data["images"]:
        i += 1
        #print(f'Reading file: {img['file_name']}  - ({i})')
        name = img['file_name']
        print(f'Reading file: {name}')
        if "width" not in img or "height" not in img:
            image = np.array(Image.open(f"new_dataset/Images/{img['file_name']}"))
            h, w = image.shape[:2]
            img["height"] = h
            img["width"] = w

    with open("new_dataset/Annotation_fixed.json", "w") as f2:
        json.dump(data, f2, indent=True)

'''
with open("new_dataset/Annotation_fixed.json") as f:
    data2 = json.load(f)
    print("JSON valid. Keys:", data2.keys())
'''


'''
{
            "id": 2672,
            "file_name": "N-260-01.jpg"
        },
'''