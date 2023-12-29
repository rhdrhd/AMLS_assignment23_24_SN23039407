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
import wandb

from torch.utils.data import Dataset
from torchvision import transforms

from torchvision.transforms.functional import to_tensor

wandb.login()
wandb.init(project="aml-final-dense")

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

lowest_val_loss = float('inf')  # Set initial loss to infinity
best_model = None  # Placeholder for the best model

# Create DenseNet model

model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=9)

train_dataset, val_dataset, test_dataset = utils.load_dataset_t2()

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle= True, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


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
            val_output = model(data)
            target = target.reshape(-1).long().cuda()
            val_loss = criterion(val_output, target)
            _, pred_label = torch.max(val_output, 1)
            num_correct += (pred_label == target).sum().item()
            num_samples += pred_label.size(0)
        print(f"val_loss:{val_loss}")
        print(f"Validation Accuracy: {100*num_correct / num_samples:.2f}")

    if val_loss < lowest_val_loss:
        lowest_val_loss = val_loss
        best_model = model.state_dict()  # Save the model parameters

    # Save the best model to a file
    torch.save(best_model, 'best_model.pth')
    wandb.log({"epoch":epoch})
    wandb.log({"train_loss":loss})
    wandb.log({"val_accuracy": 100*num_correct / num_samples})
    wandb.log({"val_loss":val_loss})
