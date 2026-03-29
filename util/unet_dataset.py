from torch.utils.data.dataset import Dataset
from skimage.io import imread
import warnings
import torch
from config import classes_color

class UNET_Dataset(Dataset):
    'Characterizes a dataset for PyTorch'
    def __init__(self, image_filenames, mask_filenames, augmentator):
        """
        Args:
            image_filenames (list of str): List of paths to the image files.
            mask_filenames (list of str): List of paths to the mask files.
            augmentator (callable): A function/transform for augmenting images and masks.
        """

        self.image_files = [imread(path) for path in image_filenames]
        self.mask_files = [imread(path) for path in mask_filenames]
        self.augment = augmentator

        #print(image_filenames[:2], end="\n")
        #print(mask_filenames[:2], end="\n")

        #for path in image_filenames:
        #    if path in mask_filenames:
        #        self.image_files =

    def __getitem__(self, index):
        """
        Generates one sample of data.

        Args:
            index (int): Index of the sample to fetch.

        Returns:
            tuple: (image, mask) where image is a torch tensor and mask is a tensor with class indices.
        """
        img, mask = self.image_files[index], self.mask_files[index]
        
        augmented = self.augment(image=img, mask=mask)
        img = augmented['image'] / 255.0
        mask = augmented['mask']

        final_mask = self._mask_to_class(mask)
        final_mask = final_mask[:, :, 0].long()
        img_tensor = torch.from_numpy(img).float()

        return img_tensor, final_mask
    
    def _mask_to_class(self, mask):
        """
        Converts mask values to class indices.

        Args:
            mask (np.ndarray): The mask to convert from RGB to ID.

        Returns:
            torch.Tensor: Mask with class indices.
        """

        mask_tensor = torch.from_numpy(mask)#.long()

        for idx, color in enumerate(classes_color):

            mask_tensor[torch.all(mask_tensor == color, dim=-1)] = idx

        return mask_tensor

    def __len__(self):
        """Returns the total number of samples in the dataset."""
        return len(self.image_files)