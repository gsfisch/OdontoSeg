import os
import torch
from typing import List, Tuple
from util.aug import create_augmentations
from config import training_config
from util.unet_dataset import UNET_Dataset

def load_images_from_folder(folder: str) -> Tuple[List[str], List[str]]:
    """
    Loads image and mask file paths from the given folder. 

    Args:
        folder (str): Path to the dataset folder.

    Returns:
        Tuple[List[str], List[str]]: Lists of image and mask file paths.
    """
    images = []
    masks = []

    images_path = os.path.join(folder, 'images')
    masks_path = os.path.join(folder, 'masks')

    if not os.path.exists(images_path) or not os.path.exists(masks_path):
        raise FileNotFoundError(f"Directories {images_path} or {masks_path} do not exist.")

    for file in os.listdir(images_path):
        if file.endswith(".png") and file in os.listdir(masks_path):
            image_file = os.path.join(images_path, file)
            mask_file = os.path.join(masks_path, file)

            images.append(image_file)
            masks.append(mask_file)

    return images, masks

def load_data(dataset_path: str) -> Tuple[List[str], List[str], List[str], List[str]]:
    """
    Loads training and validation data from the dataset path.

    Args:
        dataset_path (str): Path to the dataset directory.

    Returns:
        Tuple[List[str], List[str], List[str], List[str]]:
            Lists of training images, training masks, validation images, and validation masks.
    """
    train_path = os.path.join(dataset_path, 'train')
    valid_path = os.path.join(dataset_path, 'validation')
    test_path = os.path.join(dataset_path, 'test')

    x_train, y_train = load_images_from_folder(train_path)
    x_valid, y_valid = load_images_from_folder(valid_path)
    x_test, y_test = load_images_from_folder(test_path)

    return x_train, y_train, x_valid, y_valid, x_test, y_test

def create_data_loader(dataset: UNET_Dataset, batch_size: int, shuffle: bool, num_workers: int) -> torch.utils.data.DataLoader:
    """
    Creates a DataLoader for the given dataset.

    Args:
        dataset (UNET_Dataset): The dataset to load.
        batch_size (int): Number of samples per batch.
        shuffle (bool): Whether to shuffle the data.
        num_workers (int): Number of worker processes to use.

    Returns:
        torch.utils.data.DataLoader: Configured DataLoader.
    """
    return torch.utils.data.DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)

def get_data_generators(batch_size=training_config['batch_size'], dataset_path=training_config['dataset_path'], seed=None) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """
    Prepares training and validation DataLoaders.

    Returns:
        Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
            Training and validation DataLoaders.
    """
    #dataset_path = training_config['dataset_path']
    #batch_size = training_config['batch_size']
    val_batch_size = batch_size #training_config['val_batch_size']

    AUGMENTATIONS_TRAIN, AUGMENTATIONS_VALID = create_augmentations(seed=seed)

    x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(dataset_path)

    data_train = UNET_Dataset(x_train, y_train, AUGMENTATIONS_TRAIN)
    data_valid = UNET_Dataset(x_valid, y_valid, AUGMENTATIONS_VALID)
    data_test = UNET_Dataset(x_test, y_test, AUGMENTATIONS_VALID)

    train_loader = create_data_loader(data_train, batch_size, shuffle=True, num_workers=0)
    valid_loader = create_data_loader(data_valid, batch_size=val_batch_size, shuffle=False, num_workers=0)
    test_loader = create_data_loader(data_test, batch_size=val_batch_size, shuffle=False, num_workers=0)

    return train_loader, valid_loader, test_loader

def load_data_test(dataset_path: str) -> Tuple[List[str], List[str]]:
    """
    Loads test image and mask file paths from the dataset path.

    Args:
        dataset_path (str): Path to the dataset directory.

    Returns:
        Tuple[List[str], List[str]]: Lists of test image and mask file paths.
    """
    test_path = os.path.join(dataset_path, 'test')
    x_test, y_test = load_images_from_folder(test_path)
    print('Test Images:')
    print(len(x_test))

    return x_test, y_test

def get_test_generator() -> torch.utils.data.DataLoader:
    """
    Prepares the test DataLoader.

    Returns:
        torch.utils.data.DataLoader: Test DataLoader.
    """
    dataset_path = training_config['dataset_path']

    _, _, _, _, _, _ = load_data(dataset_path)  # Load data to initialize augmentations (if needed)
    AUGMENTATIONS_TRAIN, AUGMENTATIONS_VALID = create_augmentations()  # May not be used

    x_test, y_test = load_data_test(dataset_path)
    data_test = UNET_Dataset(x_test, y_test, AUGMENTATIONS_VALID)

    return create_data_loader(data_test, batch_size=1, shuffle=False, num_workers=0)
