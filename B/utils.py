import numpy as np
import random
import tensorflow as tf
import sys
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from skimage.transform import rotate
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



