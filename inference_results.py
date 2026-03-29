import matplotlib.pyplot as plt


original_img = plt.imread("./dataset/test/images/carcinoma_37.png")
original_mask = plt.imread("./dataset/test/masks/carcinoma_37.png")

original_img2 = plt.imread("./dataset/test/images/leucoplasia_N-103.png")
original_mask2 = plt.imread("./dataset/test/masks/leucoplasia_N-103.png")

original_img3 = plt.imread("./dataset/test/images/ploliferativas_IMG_2088.png")
original_mask3 = plt.imread("./dataset/test/masks/ploliferativas_IMG_2088.png")


# MyArch
MyArch_img = plt.imread("inference/MyArchitecture/2025-12-07_17:56:57.566/segmented_image.jpeg")
MyArch_mask = plt.imread("inference/MyArchitecture/2025-12-07_17:56:57.566/masks_image.jpeg")

MyArch_img2 = plt.imread("inference/MyArchitecture/2025-12-07_20:32:15.016/segmented_image.jpeg")
MyArch_mask2 = plt.imread("inference/MyArchitecture/2025-12-07_20:32:15.016/masks_image.jpeg")

MyArch_img3 = plt.imread("inference/MyArchitecture/2025-12-07_20:35:39.981/segmented_image.jpeg")
MyArch_mask3 = plt.imread("inference/MyArchitecture/2025-12-07_20:35:39.981/masks_image.jpeg")


# MyArch v2
MyArch_v2_img = plt.imread("inference/MyArchitecture_v2_second_try/2025-12-07_17:53:03.505/segmented_image.jpeg")
MyArch_v2_mask = plt.imread("inference/MyArchitecture_v2_second_try/2025-12-07_17:53:03.505/masks_image.jpeg")

MyArch_v2_img2 = plt.imread("inference/MyArchitecture_v2_second_try/2025-12-07_20:47:24.358/segmented_image.jpeg")
MyArch_v2_mask2 = plt.imread("inference/MyArchitecture_v2_second_try/2025-12-07_20:47:24.358/masks_image.jpeg")

MyArch_v2_img3 = plt.imread("inference/MyArchitecture_v2_second_try/2025-12-07_20:47:41.175/segmented_image.jpeg")
MyArch_v2_mask3 = plt.imread("inference/MyArchitecture_v2_second_try/2025-12-07_20:47:41.175/masks_image.jpeg")


# MyArch pvt
MyArch_pvt_img = plt.imread("inference/MyArchitecture_pvt_v2_b1__U_Net/2025-12-07_17:54:08.111/segmented_image.jpeg")
MyArch_pvt_mask = plt.imread("inference/MyArchitecture_pvt_v2_b1__U_Net/2025-12-07_17:54:08.111/masks_image.jpeg")

MyArch_pvt_img2 = plt.imread("inference/MyArchitecture_pvt_v2_b1__U_Net/2025-12-07_20:53:05.386/segmented_image.jpeg")
MyArch_pvt_mask2 = plt.imread("inference/MyArchitecture_pvt_v2_b1__U_Net/2025-12-07_20:53:05.386/masks_image.jpeg")

MyArch_pvt_img3 = plt.imread("inference/MyArchitecture_pvt_v2_b1__U_Net/2025-12-07_20:53:21.484/segmented_image.jpeg")
MyArch_pvt_mask3 = plt.imread("inference/MyArchitecture_pvt_v2_b1__U_Net/2025-12-07_20:53:21.484/masks_image.jpeg")


fig, axes = plt.subplots(5, 3, figsize=(12, 12))


# Dataset
axes[0, 0].imshow(original_img); axes[0, 0].axis('off'); axes[0, 0].text(-0.1, 0.5, 'Original', va='center', ha='right', transform=axes[0, 0].transAxes)
axes[1, 0].imshow(original_mask); axes[1, 0].axis('off'); axes[0, 0].text(-0.1, 0.5, 'Ground Truth', va='center', ha='right', transform=axes[1, 0].transAxes)

axes[0, 1].imshow(original_img2); axes[0, 1].axis('off')
axes[1, 1].imshow(original_mask2); axes[1, 1].axis('off')

axes[0, 2].imshow(original_img3); axes[0, 2].axis('off')
axes[1, 2].imshow(original_mask3); axes[1, 2].axis('off')


# MyArch
axes[2, 0].imshow(MyArch_img); axes[2, 0].axis('off'); axes[2, 0].text(-0.1, 0.5, 'swin_base_patch4_window7_224', va='center', ha='right', transform=axes[2, 0].transAxes)
axes[2, 1].imshow(MyArch_img2); axes[2, 1].axis('off')
axes[2, 2].imshow(MyArch_img3); axes[2, 2].axis('off')


# MyArch_v2
axes[3, 0].imshow(MyArch_v2_img); axes[3, 0].axis('off'); axes[3, 0].text(-0.1, 0.5, 'Tiny_vit_21m_512', va='center', ha='right', transform=axes[3, 0].transAxes)
axes[3, 1].imshow(MyArch_v2_img2); axes[3, 1].axis('off')
axes[3, 2].imshow(MyArch_v2_img3); axes[3, 2].axis('off')


# MyArch_pvt
axes[4, 0].imshow(MyArch_pvt_img); axes[4, 0].axis('off'); axes[4, 0].text(-0.1, 0.5, 'pvt_v2_b1', va='center', ha='right', transform=axes[4, 0].transAxes)
axes[4, 1].imshow(MyArch_pvt_img2); axes[4, 1].axis('off')
axes[4, 2].imshow(MyArch_pvt_img3); axes[4, 2].axis('off')

plt.tight_layout()
plt.show()
