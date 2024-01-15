import numpy as np
import random
import sys
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from skimage.transform import rotate
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import seaborn as sns
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

        if self.transform:
            image = self.transform(image)

        label = torch.LongTensor(label)
        return image, label

def evaluate_performance_metrics(true_labels, predictions, probabilities, class_names, model_name):

    # Convert lists to numpy arrays 
    if not isinstance(true_labels, np.ndarray):
        true_labels = np.array(true_labels)
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)

    # Calculating metrics
    auc = roc_auc_score(true_labels, probabilities,  multi_class='ovr', average='weighted')
    report = classification_report(true_labels, predictions, target_names=class_names, digits=6)

    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    # Plotting the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f'B/images/confusion_matrix_{model_name}.png')

    print(report)
    print(f"AUC: {auc:.4f}")



def get_mean_std(loader):
    # Vectors to store the sum and square sum of all elements in the dataset for each channel
    channel_sum, channel_sq_sum, num_batches = 0, 0, 0

    for data,_ in loader:

        channel_sum += torch.mean(data, dim=[0, 2, 3])  # Mean over batch, height, and width
        channel_sq_sum += torch.mean(data**2, dim=[0, 2, 3])  # Squared mean over batch, height, and width
        num_batches += 1
        mean = channel_sum / num_batches
        std = (channel_sq_sum / num_batches - mean**2)**0.5  # Standard deviation

    return mean, std

def load_dataset_t2(model_name="ResNet50_28_dropout", augmented=True):
    
    #load dataset from Datasets folder
    data = np.load('Datasets/pathmnist.npz')
    
    #calculate mean and std from train
    train_dataset = CustomDataset(data['train_images'], data['train_labels'],transform=transforms.Compose([transforms.ToTensor()]))
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,drop_last=True)
    mean, std = get_mean_std(train_loader)
    basic_transform_list = [
        transforms.ToTensor(),
        transforms.Normalize(mean, std)  # Replace 'mean' and 'std' with your values
    ]
    
    if model_name == "DenseNet121":
        basic_transform_list.append(transforms.Pad((1, 1, 0, 0)))
    if model_name in ("ResNet18_32","ResNet18_32_dropout","ResNet50_32","ResNet50_32_dropout"):
        basic_transform_list.append(transforms.Resize((32,32)))
    if model_name in ("ResNet18_224"):
        basic_transform_list.append(transforms.Resize((224,224)))

    transform_for_test = transforms.Compose(basic_transform_list)

    #apply extra transformation for different model selected
    aug_transform_list = basic_transform_list.copy()
    if augmented:
        aug_transform_list.append(transforms.RandomHorizontalFlip())
        aug_transform_list.append(transforms.RandomRotation(10))

    transform_for_train = transforms.Compose(aug_transform_list)

    #apply augment and normalize transform to train, val and only normalize to test
    normalize_train = CustomDataset(data['train_images'], data['train_labels'], transform=transform_for_train)
    normalize_val = CustomDataset(data['val_images'], data['val_labels'], transform= transform_for_train)
    normalize_test = CustomDataset(data['test_images'], data['test_labels'], transform= transform_for_test)

    return normalize_train, normalize_val, normalize_test


#x_train, y_train, x_val, y_val, x_test, y_test = load_and_preprocess_dataset()
#train, val, test = load_dataset_t2("ResNet18")
#train_loader = DataLoader(train, batch_size=32, shuffle=True,drop_last=True)

#for item in train_loader:
#    print(item[0].max())
#    print(item[1])
#    break

#x_train, y_train = convert_dataset_for_classical_ml(train)
#print(x_train.shape)
#print(y_train.shape)

