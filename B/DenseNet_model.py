import sys
import os
import utils


import monai
from monai.networks.nets import DenseNet121
from monai.transforms import Compose, ToTensor, Resize, ScaleIntensity
from monai.data import Dataset, DataLoader
import medmnist
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim


from torch.utils.data import Dataset
from torchvision import transforms

from torchvision.transforms.functional import to_tensor

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
x_train, y_train, x_val, y_val, x_test, y_test = utils.load_and_preprocess_dataset('task2')

x_train = x_train.reshape((-1, 28, 28, 3))
x_val = x_val.reshape((-1, 28, 28, 3))
x_test = x_test.reshape((-1, 28, 28, 3))


# Create DenseNet model

model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=9)

print(x_train.shape)
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

# Define transformations
transform = transforms.Compose([
    #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    transforms.Pad((1, 1, 0, 0))
])  # Adjust these values based on your dataset

# Create datasets
train_dataset = CustomDataset(x_train, y_train, transform=transform)
val_dataset = CustomDataset(x_val, y_val, transform= transform)
test_dataset = CustomDataset(x_test, y_test, transform= transform)


# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle= True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32)

for input, label in train_dataset:
    print(input.shape)
    print(label.shape)
    break
# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


model.to(device)
# Training loop
num_epochs = 1
for epoch in range(num_epochs):
    model.train()
    for data, target in train_loader:

        optimizer.zero_grad()
        data = data.cuda()
        output = model(data)
        target = target.reshape(-1).long().cuda()
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
    
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}")

    model.eval()
    with torch.no_grad():
        num_correct = 0
        num_samples = 0
        for data, target in val_loader:
            data = data.cuda()
            outputs = model(data)
            target = target.reshape(-1).long().cuda()
            val_loss = criterion(output, target)
            _, pred_label = torch.max(outputs, 1)
            num_correct += (pred_label == target).sum().item()
            num_samples += pred_label.size(0)
        print(f"val_loss:{val_loss}")
        print(f"Validation Accuracy: {num_correct / num_samples:.2f}")