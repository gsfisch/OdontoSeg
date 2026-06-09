import torch
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from validation_config import validation_config
import os
import ast
from util.data import get_data_generators
from util.model import make_model

from sklearn.metrics import roc_curve, auc, precision_recall_curve
from sklearn.preprocessing import label_binarize
from sklearn.metrics import average_precision_score


NUM_CLASSES = 4
LESION_CLASSES = [0, 1, 2]


label_name = {
    'efficientnet-b6_FPN': 'EfficientNet + FPN',
    'efficientnet-b6_Linknet': 'EfficientNet + LinkNet',
    'efficientnet-b6_U-Net': 'EfficientNet + U-Net',
    'resnet34_U-Net': 'ResNet34 + U-Net',
    'resnet34_Linknet': 'ResNet34 + LinkNet',
    'resnet34_FPN': 'ResNet34 + FPN',
    'resnet101_U-Net': 'ResNet101 + U-Net',
    'resnet101_Linknet': 'ResNet101 + LinkNet',
    'resnet101_FPN': 'ResNet101 + FPN',
    'SegFormer_mit_b0': 'SegFormer b0',
    'SegFormer_mit_b1': 'SegFormer b1',
    'SegFormer_mit_b2': 'SegFormer b2',
    'SegFormer_mit_b3': 'SegFormer b3',
    'SegFormer_mit_b4': 'SegFormer b4',
    'SegFormer_mit_b5': 'SegFormer b5',
    'vgg16_FPN': 'VGG16 + FPN',
    'vgg16_Linknet': 'VGG16 + LinkNet',
    'vgg16_U-Net': 'VGG16 + U-Net',
    'vit_large_patch16_224_FPN': 'ViT + FPN',
    'vit_large_patch16_224_Linknet': 'ViT + LinkNet',
    'vit_large_patch16_224_MAnet': 'ViT + MA-Net',
    'vit_large_patch16_224_U-Net': 'ViT + U-Net',
    'vit_large_patch16_224_U-Net++': 'ViT + U-Net++',
    'deit3_base_patch16_224_FPN': 'DeiT 3 + FPN',
    'deit3_base_patch16_224_Linknet': 'DeiT 3 + LinkNet',
    'deit3_base_patch16_224_MAnet': 'DeiT 3 + MA-Net',
    'deit3_base_patch16_224_U-Net': 'DeiT 3 + U-Net',
    'deit3_base_patch16_224_U-Net++': 'DeiT 3 + U-Net++',
    'swin_large_patch4_window7_224_FPN': 'Swin Transf. + FPN',
    'swin_large_patch4_window7_224_Linknet': 'Swin Transf. + LinkNet',
    'swin_large_patch4_window7_224_MAnet': 'Swin Transf. + MA-Net',
    'swin_large_patch4_window7_224_U-Net': 'Swin Transf. + U-Net',
    'swin_large_patch4_window7_224_U-Net++': 'Swin Transf. + U-Net++',
    '': '',
    '': '',
    '': '',
    '': '',
    '': '',
    '': '',
    '': '',
    '': '',
    '': '',
    '': '',
    '': '',
    '': '',
}


skip_names = ['old_dataset', 'resnet101_Linknet_batch8']

model_directory_path = validation_config['model_directory_path']
configs_file_name = validation_config['configs_file_name']
model_file_name = validation_config['model_file_name']

for model_name in os.listdir('./models'):
    if model_name in skip_names:
        continue

    print(model_name)

    model_directory_path = f'models/{model_name}'
    configs_file_name = 'config.txt'
    model_file_name = f'{model_name}.pth'



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


    # get data generators
    _, _, test_generator = get_data_generators(batch_size=2, dataset_path='/home/fisch/Documents/OdontoSeg/datasets/Dataset_Imagens_Clinicas_V2.0')


    with torch.no_grad():

        total_batches = len(test_generator)
        loop = tqdm(enumerate(test_generator), total=total_batches, desc='PR')

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



    # Concatenate
    all_probs = np.concatenate(all_probs, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    print("Probabilities shape:", all_probs.shape)
    print("Targets shape:", all_targets.shape)

    print(all_targets[:4])

    # One-hot encoding
    all_targets_bin = label_binarize(
        all_targets,
        classes=np.arange(NUM_CLASSES)
    )

    print("Binary targets shape:", all_targets_bin.shape) # [Pixels, classes]


    # =========================================================
    # ROC PER LESION CLASS
    # =========================================================

    precision = dict()
    recall = dict()
    #pr_auc = dict()
    ap = dict()

    for i in LESION_CLASSES:

        precision[i], recall[i], _ = precision_recall_curve(
            all_targets_bin[:, i],
            all_probs[:, i]
        )

        '''
        pr_auc[i] = auc(
            recall[i],
            precision[i],
        )
        '''

        # average precision
        ap[i] = average_precision_score(
            all_targets_bin[:, i],
            all_probs[:, i]
        )


    # =========================================================
    # MACRO-AVERAGE PR
    # =========================================================

    # all Recall points
    all_recall = np.unique(
        np.concatenate(
            [recall[i] for i in LESION_CLASSES]
        )
    )

    # mean Precision
    mean_precision = np.zeros_like(all_recall)

    for i in LESION_CLASSES:

        mean_precision += np.interp(
            all_recall,
            recall[i],
            precision[i]
        )

    # average
    mean_precision /= len(LESION_CLASSES)

    # macro PR
    recall["macro"] = all_recall
    precision["macro"] = mean_precision

    '''
    pr_auc["macro"] = auc(
        recall["macro"],
        precision["macro"],
    )
    '''

    ap["macro"] = np.mean([ap[i] for i in LESION_CLASSES])

    precision["micro"], recall["micro"], _ = precision_recall_curve(
        all_targets_bin[:, LESION_CLASSES].ravel(),
        all_probs[:, LESION_CLASSES].ravel()
    )

    ap["micro"] = average_precision_score(
        all_targets_bin[:, LESION_CLASSES],
        all_probs[:, LESION_CLASSES],
        average="micro"
    )

    '''
    for i in LESION_CLASSES:
        # Look at distribution of predicted probabilities for that class
        plt.hist(all_probs[:, i], bins=100)
        plt.title(f"Class {i} probability histogram")
        plt.show()
    '''

    # Plot
    plt.figure(figsize=(8, 8))

    colors = ["red", "yellow", "green"]

    for i, color in zip(LESION_CLASSES, colors):

        classes = ['MMN', 'OPMD', 'PL']

        plt.plot(
            recall[i],
            precision[i],
            color=color,
            lw=2,
            #label=f"{classes[i]} lesion (AUC = {pr_auc[i]:.4f})"
            label=f"{classes[i]} lesion (AP = {ap[i]:.4f})"
        )
    '''
    # macro-average AP
    plt.plot(
        recall["macro"],
        precision["macro"],
        color="black",
        linestyle="--",
        lw=3,
        label=f"Macro-average (AP = {ap['macro']:.4f})"
    )


    # micro-average AP
    plt.plot(
        recall["micro"],
        precision["micro"],
        color="black",
        linestyle="--",
        lw=3,
        label=f"Micro-average (AP = {ap['micro']:.4f})"
    )
    '''

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title(f"PR Curves - {label_name[model_name]}")

    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./PR_curves/{model_file_name[:-4]}_PR_curve.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"./PR_curves_jpeg/{model_file_name[:-4]}_PR_curve.jpeg", dpi=300, bbox_inches='tight')
    #plt.show()



    print("\nPer-class AP:")

    for i in LESION_CLASSES:
        print(f"Lesion Class {i}: {ap[i]:.4f}")

    print(f"\nMacro-average AP: {ap['macro']:.4f}")
