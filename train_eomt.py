'''
from transformers import EomtConfig, EomtForUniversalSegmentation


# Create model configs
config = EomtConfig(
    image_size=512,
    patch_size=16,
    num_channels=3,
    num_labels=4,

    # Transformer backbone
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,

    # Dropout / regularization
    hidden_dropout_prob=0.1,
    attention_probs_dropout_prob=0.1,

    # Labels
    id2label={
        0: "class_0",
        1: "class_1",
        2: "class_2",
        3: "class_3",
    },
    label2id={
        "class_0": 0,
        "class_1": 1,
        "class_2": 2,
        "class_3": 3,
    },
)

model = EomtForUniversalSegmentation(config)




'''

from transformers import EomtForUniversalSegmentation, EomtImageProcessor
import torch
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from torchvision import transforms
import torch.nn as nn


class EomtSegmentationHead(nn.Module):
    def __init__(self, hidden_dim, num_classes, patch_size):
        super().__init__()
        self.num_classes = num_classes
        self.patch_size = patch_size

        self.classifier = nn.Conv2d(
            hidden_dim,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):
        # x: [B, N, D]
        B, N, D = x.shape
        H = W = int(N ** 0.5)

        x = x.permute(0, 2, 1)#.reshape(B, D, H, W)
        x = x.reshape(1, -1, 1, 1)
        x = self.classifier(x)

        # Upsample from patch grid → full resolution
        x = nn.functional.interpolate(
            x,
            scale_factor=self.patch_size,
            mode="bilinear",
            align_corners=False
        )

        return x


image_dir = "/home/fisch/Documents/OdontoSeg/dataset/train/images"
mask_dir = "/home/fisch/Documents/OdontoSeg/dataset/train/masks"
#dataset/train/images
#/home/fisch/Documents/OdontoSeg/dataset/train/images

image_paths = sorted([
    os.path.join(image_dir, f)
    for f in os.listdir(image_dir)
    #if f.endswith((".png", ".jpg"))
])

mask_paths = sorted([
    os.path.join(mask_dir, f)
    for f in os.listdir(mask_dir)
    #if f.endswith(".png")
])



class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, processor=1):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.processor = processor

    def __getitem__(self, idx):
        # Load the image and mask
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])

        # Preprocess the image using the processor
        #encoded = self.processor(
        #    images=image,
        #    segmentation_maps=mask,  # Pass the mask as well for processing
        #    return_tensors="pt"
        #)

        #print(encoded)

        # Here, we manually extract the image and mask
        #pixel_values = encoded["pixel_values"].squeeze(0)  # Get the image tensor
        #labels = encoded["class_labels"].squeeze(0)              # Get the mask tensor

        transform = transforms.ToTensor()

        # Convert PIL image to PyTorch tensor
        pixel_values = transform(image)
        labels = transform(mask)
        

        # Return both image and mask tensors as a dictionary
        return {
            "pixel_values": pixel_values,  # Image tensor
            "labels": labels               # Mask tensor (labels)
        }

    def __len__(self):
        return len(self.image_paths)

processor = EomtImageProcessor(
    do_resize=True,
    size={"shortest_edge": 512, "longest_edge": 512},
    do_rescale=True,
    rescale_factor=1 / 255.0,
    do_normalize=True,
    image_mean=[0.485, 0.456, 0.406],
    image_std=[0.229, 0.224, 0.225],
    do_reduce_labels=False,  # don't modify the labels
)

train_dataset = SegmentationDataset(image_paths=image_paths, mask_paths=mask_paths) #processor=processor
train_loader = DataLoader(
    train_dataset,
    batch_size=2,     # 512×512 is heavy
    shuffle=True,
    num_workers=4
)

model = EomtForUniversalSegmentation.from_pretrained(
    "tue-mps/ade20k_semantic_eomt_large_512",
    num_labels=4,
    ignore_mismatched_sizes=True
)

print(type(model))


device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

optimizer = AdamW(
    model.parameters(),
    lr=1e-4,
    weight_decay=1e-4
)

model.train()
for epoch in range(100):
    for batch in train_loader:
        pixel_values = batch["pixel_values"].to(device)
        labels = batch["labels"].to(device)

        outputs = model(
            pixel_values=pixel_values,
            labels=labels
        )

        #print(vars(outputs))
        criterion = nn.CrossEntropyLoss(ignore_index=255)

        seg_head = EomtSegmentationHead(hidden_dim=2312192, num_classes=4, patch_size=16).to(device)
        logits = seg_head(outputs.last_hidden_state)
        # logits: [B, 4, 512, 512]

        print(logits.shape)

        loss = criterion(logits, batch["labels"])

        loss = outputs.loss
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

    print(f"Epoch {epoch} | loss = {loss.item():.4f}")