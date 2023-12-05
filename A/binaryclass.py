import numpy as np
import os
from matplotlib import pyplot as plt
import wandb
print("Current Working Directory:", os.getcwd())
#the code is tested on vscode interactive window


#wandb.init(project="aml-final")

data = np.load('../Datasets/pneumoniamnist.npz')
#print(list(data.keys()))
array1 = data['val_images'] 
#plt.imshow(array1[0], cmap='gray')
#plt.show()


import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import torch.optim as optim


# Convert to PyTorch tensors
train_images = torch.Tensor(data['train_images']).unsqueeze(1) # Add channel dimension
val_images = torch.Tensor(data['val_images']).unsqueeze(1)
test_images = torch.Tensor(data['test_images']).unsqueeze(1)
train_labels = torch.FloatTensor(data['train_labels'])
val_labels = torch.FloatTensor(data['val_labels'])
test_labels = torch.FloatTensor(data['test_labels'])

# Create datasets
train_dataset = TensorDataset(train_images, train_labels)
val_dataset = TensorDataset(val_images, val_labels)
test_dataset = TensorDataset(test_images, test_labels)

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,drop_last=True)
val_loader = DataLoader(val_dataset, batch_size=64, drop_last=True)
test_loader = DataLoader(test_dataset, batch_size=64)

#check batch size
#for i, batch in enumerate(train_loader):
   #print(i, batch[0].shape)
#print(train_loader.dataset.tensors[0].shape)

#Simple model with linear layers
class Net(nn.Module):
  def __init__(self):
    super(Net,self).__init__()
    self.fc1 = nn.Linear(784,128)
    self.fc2 = nn.Linear(128,64)
    self.fc3 = nn.Linear(64,1)
  def forward(self,x):
    x = self.fc1(x)
    x = torch.relu(x)
    x = torch.relu(self.fc2(x))
    x = torch.sigmoid(self.fc3(x))
    return x

input = torch.randn(64,784)
model = Net()
output = model(input)


# Initialize the model, loss function, and optimizer
model = Net()
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

# Training loop
for epoch in range(50):  # Number of epochs
    for images, labels in train_loader:
        
        outputs = model(images.view(-1, 784))

        loss = criterion(outputs, labels)
        optimizer.zero_grad()
        
        loss.backward()
        optimizer.step()

    # Validation step
    with torch.no_grad():
        correct = 0
        total = 0
        for images, labels in val_loader:
            outputs = model(images.view(-1, 784))
            
            predicted = torch.round(outputs)
            #print(predicted)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
        #wandb.log({"val_accuracy": correct / total})
        print(f'Epoch {epoch+1}, Validation Accuracy: {100*correct / total}%')

#wandb.finish()