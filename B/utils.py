import numpy as np
import random
import sys
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from skimage.transform import rotate
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
import os


# Function to rotate an image
def random_rotate(image):
    # Random angle from -10 to 10 degrees
    angle = random.uniform(-10, 10)
    return rotate(image, angle, resize=False, mode='edge')


def add_noise(image, noise_type="gaussian"):
    """
    Add random noise to an image.
    Types of noise can include 'gaussian', 'salt_pepper', etc.
    """
    if noise_type == "gaussian":
        row, col = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col))
        noisy_image = np.clip(image + gauss, 0, 1)  # Ensure values are within [0, 1]
        return noisy_image
    elif noise_type == "salt_pepper":
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)

        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[tuple(coords)] = 1

        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper))
                  for i in image.shape]
        out[tuple(coords)] = 0
        return out

    # Add more noise types as needed
    else:
        raise ValueError("Unknown noise type")
    
def normalize_data(data):
    scaler = MinMaxScaler()
    normalized_data = scaler.fit_transform(data)
    return normalized_data

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        # Convert to tensor
        image = to_tensor(image)

        if self.transform:
            image = self.transform(image)

        return image, label



def load_dataset_new(dataset):
    # Create datasets
        
    if dataset == "task1":
        data = np.load('Datasets/pneumoniamnist.npz')
    elif dataset == "task2":
        data = np.load('Datasets/pathmnist.npz')
    else:
        print('Please add dataset in Datasets folder')

    train_dataset = CustomDataset(data['train_images'], data['train_labels'])
    val_dataset = CustomDataset(data['val_images'], data['val_labels'])
    test_dataset = CustomDataset(data['test_images'], data['test_labels'])

    return train_dataset, val_dataset, test_dataset



def get_mean_std(loader):
    # Vectors to store the sum and square sum of all elements in the dataset for each channel
    channel_sum, channel_sq_sum, num_batches = 0, 0, 0

    for batch in loader:
        #print(len(batch))
        data = batch[0]
        channel_sum += torch.mean(data, dim=[0, 2, 3])  # Mean over batch, height, and width
        channel_sq_sum += torch.mean(data**2, dim=[0, 2, 3])  # Squared mean over batch, height, and width
        num_batches += 1
        mean = channel_sum / num_batches
        std = (channel_sq_sum / num_batches - mean**2)**0.5  # Standard deviation

    return mean, std

def load_dataset_t2():
    
    data = np.load('Datasets/pathmnist.npz')
    
    train_dataset = CustomDataset(data['train_images'], data['train_labels'])
    val_dataset = CustomDataset(data['val_images'], data['val_labels'])
    test_dataset = CustomDataset(data['test_images'], data['test_labels'])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,drop_last=True)
    mean, std = get_mean_std(train_loader)
    
    #calculate mean and std from train
    normalize_pad_transform = transforms.Compose([
        transforms.Normalize(mean, std),
        transforms.Pad((1, 1, 0, 0))
    ])  
    
    normalize_train = CustomDataset(data['train_images'], data['train_labels'], transform=normalize_pad_transform)
    normalize_val = CustomDataset(data['val_images'], data['val_labels'], transform= normalize_pad_transform)
    normalize_test = CustomDataset(data['test_images'], data['test_labels'], transform= normalize_pad_transform)

    return normalize_train, normalize_val, normalize_test

#dead one
def load_and_preprocess_dataset(dataset):
    
    if dataset == "task1":
        data = np.load('Datasets/pneumoniamnist.npz')
    elif dataset == "task2":
        data = np.load('Datasets/pathmnist.npz')
    else:
        print('Please add dataset in Datasets folder')

    #extract train, validation and test from data
    train_images = data['train_images']
    train_labels = data['train_labels']
    val_images = data['val_images']
    val_labels = data['val_labels']
    test_images = data['test_images']
    test_labels = data['test_labels']

    dataset = [train_images, train_labels, val_images, val_labels, test_images, test_labels]
    for i in range(len(dataset)):
        dataset[i] = dataset[i].astype('float32').reshape(dataset[i].shape[0], -1)
        dataset[i] = normalize_data(dataset[i])

    return dataset[:]

#x_train, y_train, x_val, y_val, x_test, y_test = load_and_preprocess_dataset()
#train, val, test = load_dataset_t2()
#train_loader = DataLoader(train, batch_size=32, shuffle=True,drop_last=True)

#for item in train_loader:
#    print(item[0].shape)
#    print(item[1])
#    break
    


