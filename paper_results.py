import matplotlib.pyplot as plt
import cv2
import numpy as np
import os



def blendImage(original_path, image_path):
    """Cria uma imagem misturada entre a original e a máscara."""
    original_img = cv2.imread(original_path)
    mask  = cv2.imread(image_path)
    
    blue_color = [255, 0, 0]
    blue_mask = cv2.inRange(mask, np.array(blue_color), np.array(blue_color))
    non_blue_mask = cv2.bitwise_not(blue_mask)

    alpha = 0.6
    blended_image = cv2.addWeighted(original_img, alpha, mask, 1 - alpha, 0)
    blended_image = cv2.bitwise_and(blended_image, blended_image, mask=non_blue_mask)
    blended_image += cv2.bitwise_and(original_img, original_img, mask=blue_mask)

    cv2.imwrite(image_path, blended_image)





models = [
    #'images',
    #'masks',
    'GT',
    'vit_large_patch16_224_FPN',
    'swin_large_patch4_window7_224_MAnet',
    'swin_s3_tiny_224_U-Net',
    'swinv2_large_window12to16_192to256_U-Net',
    'deit3_base_patch16_224_U-Net',
    'caformer_b36_U-Net++',
    'SegFormer_mit_b1',
    'UNETR',
    'SwinUNETR'
]

names = [
    #'Images',
    #'Masks',
    'GT',
    'ViT + FPN',
    'Swin + MA-net',
    'Swin s3 tiny',
    'Swin V2 large',
    'DeiT 3 + U-Net',
    'CAFormer + U-Net++',
    'SegFormer',
    'UNETR',
    'SwinUNETR'
]

images = [
    'carcinoma_37.png',
    'carcinoma_31547_2.png',
    'leucoplasia_10.png',
    'leucoplasia_N-103.png',
    'ploliferativas_IMG_2088.png',
    'ploliferativas_IMG_2089.png'
]

for image in os.listdir('/home/fisch/Documents/OdontoSeg/teaser_dataset (copy)/images/'):
    original = f'/home/fisch/Documents/OdontoSeg/teaser_dataset (copy)/images/{image}'
    mask = f'/home/fisch/Documents/OdontoSeg/teaser_dataset (copy)/masks/{image}'
    

    blendImage(original, mask)



exit()
fig, axes = plt.subplots(len(models), len(images), figsize=(12, 22))

for i, name in enumerate(models):
    for j, image in enumerate(images):
        axes[i, j].imshow(plt.imread(f'./segmentation_results/{name}/{image}'))
        axes[i, j].axis('off')


        if j == 0:
            axes[i, j].text(-0.1, 0.5, names[i], va='center', ha='right', transform=axes[i, j].transAxes)
            #axes[i, j].set_ylabel(name, rotation=0, labelpad=40, va='center')

#fig.subplots_adjust(wspace=0)
#plt.subplots_adjust(wspace=-0.5)


plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=-0.89, hspace=0.15)
#fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0.02)
plt.show()

#plt.tight_layout()
#plt.subplots_adjust(left=0, right=1)
#plt.savefig('oi.png', bbox_inches='tight', pad_inches=0)
