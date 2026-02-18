import matplotlib.pyplot as plt
import requests
import torch
from PIL import Image
from transformers import EomtForUniversalSegmentation, AutoImageProcessor


model_id = "tue-mps/ade20k_semantic_eomt_large_512"
processor = AutoImageProcessor.from_pretrained(model_id)
model = EomtForUniversalSegmentation.from_pretrained(model_id)

#image = Image.open(requests.get("http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)
image = Image.open("dataset/train/images/carcinoma_1.png")

inputs = processor(
    images=image,
    return_tensors="pt",
)

with torch.inference_mode():
    outputs = model(**inputs)

# Prepare the original image size in the format (height, width)
target_sizes = [(image.height, image.width)]

# Post-process the model outputs to get final segmentation prediction
preds = processor.post_process_semantic_segmentation(
    outputs,
    target_sizes=target_sizes,
)

# Visualize the segmentation mask
plt.imshow(preds[0])
plt.axis("off")
plt.title("Semantic Segmentation")
plt.show()
