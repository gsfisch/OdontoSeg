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
    'caformer_b36_FPN': 'CAFormer + FPN',
    'caformer_b36_Linknet': 'CAFormer + LinkNet',
    'caformer_b36_MAnet': 'CAFormer + MA-Net',
    'caformer_b36_U-Net': 'CAFormer + U-Net',
    'caformer_b36_U-Net++': 'CAFormer + U-Net++',
    'inceptionv4_FPN': 'Inception V4 + FPN',
    'inceptionv4_Linknet': 'Inception V4 + LinkNet',
    'inceptionv4_U-Net': 'Inception V4 + U-Net',
    'resnet101_FPN': 'ResNet101 + FPN',
    'resnet101_Linknet': 'ResNet101 + LinkNet',
    'resnet101_U-Net': 'ResNet101 + U-Net',
    'resnet34_FPN': 'ResNet34 + FPN',
    'resnet34_Linknet': 'ResNet34 + LinkNet',
    'MedFormer': 'MedFormer',
    'TransUNet': 'TransUNet',
    'UNETR_64': 'UNETR',
    'SwinUNETR': 'SwinUNETR',
}


skip_names = ['old_dataset', 'resnet101_FPN_batch8_rangerlars', 'resnet101_U-Net_batch8_rangerlars', 'UNETR_old']

NUM_CLASSES = 4
LESION_CLASSES = [0, 1, 2]


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


    # One hot encoding
    all_targets_bin = label_binarize(
        all_targets,
        classes=np.arange(NUM_CLASSES)
    )


    # ROC per lesion
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


    # Average ROC

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


    # Plot
    plt.figure(figsize=(8, 8))

    colors = ["r", "y", "g"]
    shape = ['-' , '--', ':']

    for i in range(len(colors)):
        classes = ['MMN', 'OPMD', 'PL']

        plt.plot(
            fpr[i],
            tpr[i],
            f'{shape[i]}{colors[i]}',
            lw=2,
            label=f"{classes[i]} lesion (AUC = {roc_auc[i]:.4f})"
        )


    # random baseline
    plt.plot(
        [0, 1],
        [0, 1],
        "k:",
        lw=1
    )

    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])

    plt.xlabel("False Positive Rate", fontsize=18)
    plt.ylabel("True Positive Rate", fontsize=18)

    plt.title(f"ROC Curves - {label_name[model_name]}", fontsize=24)

    plt.legend(loc="lower right")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"./ROC_curves/{model_file_name[:-4]}_ROC_curve.png", dpi=300, bbox_inches='tight')
    plt.savefig(f"./ROC_curves_jpeg/{model_file_name[:-4]}_ROC_curve.jpeg", dpi=300, bbox_inches='tight')


    print("\nPer-class AUC:")

    for i in LESION_CLASSES:
        print(f"Lesion Class {i}: {roc_auc[i]:.4f}")
