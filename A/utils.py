import numpy as np
import random
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from skimage.transform import rotate
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import seaborn as sns

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None, target_transform=None):
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

        
        label = torch.FloatTensor(label)

        return image, label


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

def evaluate_performance_metrics(true_labels, predictions, class_names, model_name):

    # Convert lists to numpy arrays 
    if not isinstance(true_labels, np.ndarray):
        true_labels = np.array(true_labels)
    if not isinstance(predictions, np.ndarray):
        predictions = np.array(predictions)

    # Calculating metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1 = f1_score(true_labels, predictions)
    auc = roc_auc_score(true_labels, predictions)

    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)

    # Plotting the confusion matrix
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig(f'TaskA_Confusion_Matrix_{model_name}.png')

    # Print metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"Precision: {precision:.4f}")
    #print(f"F1 Score: {f1:.4f}")
    



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


def load_dataset_t1():
    
    data = np.load('Datasets/pneumoniamnist.npz')
    
    train_dataset = CustomDataset(data['train_images'], data['train_labels'],transform=transforms.Compose([transforms.ToTensor()]))
    val_dataset = CustomDataset(data['val_images'], data['val_labels'])
    test_dataset = CustomDataset(data['test_images'], data['test_labels'])

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,drop_last=True)

    mean, std = get_mean_std(train_loader)
    
    #calculate mean and std from train
    normalize_pad_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std)

    ])  
    
  
    normalize_train = CustomDataset(data['train_images'], data['train_labels'], transform=normalize_pad_transform)
    normalize_val = CustomDataset(data['val_images'], data['val_labels'], transform= normalize_pad_transform)
    normalize_test = CustomDataset(data['test_images'], data['test_labels'], transform= normalize_pad_transform)

    return normalize_train, normalize_val, normalize_test

def convert_dataset_for_classical_ml(dataset):
    data_list = []
    label_list = []

    for data, label in dataset:

        data_list.append(data.numpy())
        label_list.append(label.numpy())

    data_array = np.asarray(data_list)
    label_array = np.asarray(label_list)

    return data_array, label_array


#x_train, y_train, x_val, y_val, x_test, y_test = load_and_preprocess_dataset()


#train, val, test = load_dataset_t1()
#for one in train:
#    print(one[0].dtype)
#    print(one[1].dtype)
#    break
#x_train, y_train = convert_dataset_for_classical_ml(train)
#print(x_train.shape)
#print(y_train.shape)


