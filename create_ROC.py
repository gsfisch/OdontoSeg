import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from validation_config import validation_config
import os
import ast
from util.data import get_data_generators
from util.model import make_model

from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize


# =========================================================
# SETTINGS
# =========================================================

NUM_CLASSES = 4                  # 0=background, 1-3=lesions
LESION_CLASSES = [0, 1, 2]


model_directory_path = validation_config['model_directory_path']
configs_file_name = validation_config['configs_file_name']
model_file_name = validation_config['model_file_name']


# Read training configurations
training_config = {}
with open(os.path.join(model_directory_path, configs_file_name), 'r') as configs_file:
    training_config = ast.literal_eval(configs_file.read())


# Initialize and load model
if training_config['library'] == 'smp':
    model = make_model(training_config['encoder'], training_config['architecture'],
                            training_config['classes'], library='smp',
                            decoder_channels=training_config['decoder_channels'], encoder_depth=training_config['encoder_depth'],
                            freeze_encoder=training_config['freeze_encoder'], need_wrapper=training_config['need_wrapper']).cuda()

else:    
    model = make_model(training_config['encoder'], training_config['architecture'], 
                       classes=training_config['classes'], library=training_config['library'],
                       decoder_channels=training_config['decoder_channels'], encoder_depth=training_config['encoder_depth'],
                       encoder_params=training_config['encoder_params'], head_upsampling=training_config['head_upsampling'],
                       freeze_encoder=training_config['freeze_encoder'], need_wrapper=training_config['need_wrapper']).cuda()


model.load_state_dict(torch.load(os.path.join(model_directory_path, model_file_name), weights_only="True"))
model.eval()


all_probs = []
all_targets = []


# =========================================================
# INFERENCE LOOP
# =========================================================

# get data generators
_, _, test_generator = get_data_generators(batch_size=8, dataset_path='/home/fisch/Documents/OdontoSeg/datasets/Dataset_Imagens_Clinicas_V2.0')


with torch.no_grad():

    total_batches = len(test_generator)
    loop = tqdm(enumerate(test_generator), total=total_batches, desc='ROC')

    for batch_idx, (images, masks) in loop:

        images = images.permute(0, 3, 1, 2).cuda() # [B, C, H, W]
        masks = masks.cuda() # [B, H, W]

        # logits shape:
        # [B, C, H, W]
        logits = model(images)
        print(f'{logits.shape=}')

        # probabilities
        probs = torch.softmax(logits, dim=1)

        probs = probs.cpu().numpy()
        masks = masks.cpu().numpy()

        # -------------------------------------------------
        # reshape probabilities
        # [B,C,H,W] -> [N_pixels,C]
        # -------------------------------------------------

        probs = np.transpose(probs, (0, 2, 3, 1))
        probs = probs.reshape(-1, NUM_CLASSES)

        # masks:
        # [B,H,W] -> [N_pixels]
        masks = masks.reshape(-1)

        all_probs.append(probs)
        all_targets.append(masks)


# =========================================================
# CONCATENATE ALL DATA
# =========================================================

all_probs = np.concatenate(all_probs, axis=0)
all_targets = np.concatenate(all_targets, axis=0)

print("Probabilities shape:", all_probs.shape)
print("Targets shape:", all_targets.shape)


# =========================================================
# BINARIZE TARGETS
# =========================================================

all_targets_bin = label_binarize(
    all_targets,
    classes=np.arange(NUM_CLASSES)
)

# shape:
# [N_pixels, NUM_CLASSES]


# =========================================================
# ROC PER LESION CLASS
# =========================================================

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in LESION_CLASSES:

    fpr[i], tpr[i], _ = roc_curve(
        all_targets_bin[:, i],
        all_probs[:, i]
    )

    roc_auc[i] = auc(
        fpr[i],
        tpr[i]
    )


# =========================================================
# MACRO-AVERAGE ROC
# =========================================================

# collect all FPR points
all_fpr = np.unique(
    np.concatenate(
        [fpr[i] for i in LESION_CLASSES]
    )
)

# mean TPR
mean_tpr = np.zeros_like(all_fpr)

for i in LESION_CLASSES:

    mean_tpr += np.interp(
        all_fpr,
        fpr[i],
        tpr[i]
    )

# average
mean_tpr /= len(LESION_CLASSES)

# macro ROC
fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr

roc_auc["macro"] = auc(
    fpr["macro"],
    tpr["macro"]
)


# =========================================================
# PLOT
# =========================================================

plt.figure(figsize=(8, 8))

colors = ["red", "yellow", "green"]

for i, color in zip(LESION_CLASSES, colors):

    classes = ['MMN', 'OPMD', 'PL']

    plt.plot(
        fpr[i],
        tpr[i],
        color=color,
        lw=2,
        label=f"{classes[i]} lesion (AUC = {roc_auc[i]:.4f})"
    )

# macro-average curve
plt.plot(
    fpr["macro"],
    tpr["macro"],
    color="black",
    linestyle="--",
    lw=3,
    label=f"Macro-average (AUC = {roc_auc['macro']:.4f})"
)

# random baseline
plt.plot(
    [0, 1],
    [0, 1],
    "k:",
    lw=1
)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")

plt.title("ROC Curves for Lesion Classes")

plt.legend(loc="lower right")
plt.grid(True)

plt.tight_layout()
plt.show()


# =========================================================
# PRINT RESULTS
# =========================================================

print("\nPer-class AUC:")

for i in LESION_CLASSES:
    print(f"Lesion Class {i}: {roc_auc[i]:.4f}")

print(f"\nMacro-average AUC: {roc_auc['macro']:.4f}")