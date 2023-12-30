import numpy as np
import random

#fix random seeds to increase reproducibility
random.seed(23)
np.random.seed(23)

import utils
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader

import torch
import torch.nn as nn
from skorch import NeuralNetClassifier
from skorch.helper import predefined_split
import optuna
from torchsummary import summary


#select gpu training as priority
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The device used for training is", device)


class CustomCNN(nn.Module):
    def __init__(self,conv1_neuron=8, conv2_neuron=16):
        super(CustomCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(1, conv1_neuron, kernel_size=3)  # Assuming input channels = 1
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(conv1_neuron, conv2_neuron, kernel_size=2)

        # Fully connected (Dense) layers
        self.fc1 = nn.Linear(conv2_neuron * 12 * 12, 16)  # Adjust the input features
        self.fc2 = nn.Linear(16, 1)

        # Dropout layers
        self.dropout = nn.Dropout(0.25)

    def forward(self, x):
        # Apply first convolutional layer, followed by pooling and dropout
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)

        # Apply second convolutional layer and dropout
        x = F.relu(self.conv2(x))
        x = self.dropout(x)

        # Flatten the output for the dense layers
        x = x.view(x.shape[0], -1)  # Adjust the flattening to match the output of the last conv layer

        # Apply first dense layer
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Apply second dense layer (output layer)
        x = torch.sigmoid(self.fc2(x))

        return x



def train_customCNN(num_epochs, lr):
    #load wandb
    #wandb.login()
    #wandb.init(project="aml-final-cnn_binary")
    model = CustomCNN()
    # Create DataLoaders
    train_dataset, val_dataset, test_dataset = utils.load_dataset_t1()

    train_loader = DataLoader(train_dataset, batch_size= 64, shuffle=True,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle= True, drop_last=True)

    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    #set placeholder for best model
    lowest_val_loss = float('inf')
    highest_val_accuracy = 0
    best_model_val_loss = None  
    best_model_val_acc = None

    model.to(device)
    # Training loop

    for epoch in range(num_epochs):
        model.train()
        for data, target in train_loader:

            optimizer.zero_grad()
            data = data.cuda()
            output = model(data)
            target = target.float().cuda()
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
                target = target.cuda()
                val_loss = criterion(val_output, target)
                pred_label = torch.round(val_output)
                num_correct += (pred_label == target).sum().item()
                num_samples += pred_label.size(0)
                
            val_accuracy = 100*num_correct / num_samples
            print(f"val_loss:{val_loss}")
            print(f"Validation Accuracy: {val_accuracy:.2f}")

        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            best_model_val_loss = model.state_dict()
            torch.save(best_model_val_loss, 'A/pretrained_weights_customCNN/best_model_val_loss.pth')

        if val_accuracy > highest_val_accuracy:
            highest_val_accuracy = val_accuracy
            best_model_val_acc = model.state_dict()
            torch.save(best_model_val_acc, 'A/pretrained_weights_customCNN/best_model_val_acc.pth')

        #log performance on wandb platform
        #wandb.log({"epoch":epoch+1})
        #wandb.log({"train_loss":loss})
        #wandb.log({"val_accuracy": val_accuracy})
        #wandb.log({"val_loss":val_loss})

def test_customCNN():
    model = CustomCNN()
    # Create DataLoaders
    train_dataset, val_dataset, test_dataset = utils.load_dataset_t1()

    test_loader = DataLoader(test_dataset, batch_size=64, shuffle= True, drop_last=True)
    model.load_state_dict(torch.load("A/pretrained_weights_customCNN/best_model_val_acc.pth"))
    model.to(device)
    model.eval()

    criterion = nn.BCELoss()

    true_labels = []
    predictions = []

    with torch.no_grad():
        num_correct = 0
        num_samples = 0
        for data, target in test_loader:
            data = data.cuda()
            test_output = model(data)
            target = target.cuda()
            test_loss = criterion(test_output, target)
            pred_label = torch.round(test_output)
            predictions.extend(pred_label.cpu().numpy())
            true_labels.extend(target.cpu().numpy())
            num_correct += (pred_label == target).sum().item()
            num_samples += pred_label.size(0)

        
        classes = ["1","2"]
        #roc_auc = roc_auc_score(true_labels, predictions)
        utils.plot_confusion_matrix(true_labels, predictions, classes, 'customCNN')
        print(f"test_loss:{test_loss}")
        print(f"Test Accuracy: {num_correct / num_samples:.2f}")
        #print(f"roc auc score: {roc_auc}")


model = CustomCNN().cuda()
summary(model, (1,28,28))

