import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import A.utils as utils
from torchsummary import summary

#fix random seeds to increase reproducibility
random.seed(23)
np.random.seed(23)

#select gpu training as priority
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The device used for training is", device)


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding = 1)  # Assuming input channels = 1
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding = 1)

        # Fully connected layers
        self.fc1 = nn.Linear( 64 * 7 * 7, 64)
        self.fc2 = nn.Linear(64, 1)

        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2)

        # Dropout layer
        self.dropout= nn.Dropout(0.3)

    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)

        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)

        x = x.view(x.shape[0],-1)

        x = F.relu(self.fc1(x))
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

    best_epoch = -1
    early_stop_thresh = 20

    lowest_val_loss = float('inf')
    #highest_val_accuracy = 0
    best_model_val_loss = None  
    #best_model_val_acc = None
    train_loss_list = []
    val_loss_list = []
    

    model.to(device)
    # Training loop

    for epoch in range(num_epochs):
        total_train_loss_epoch = 0
        total_val_loss_epoch = 0
        model.train()
        for data, target in train_loader:

            optimizer.zero_grad()
            data = data.cuda()
            output = model(data)
            target = target.float().cuda()
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            total_train_loss_epoch += loss
        
        train_loss_epoch_avg=  total_train_loss_epoch/len(train_loader)
        train_loss_list.append(train_loss_epoch_avg.detach().cpu())
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
                total_val_loss_epoch += val_loss
            
            val_loss_epoch_avg=  total_val_loss_epoch/len(val_loader)
            val_loss_list.append(val_loss_epoch_avg.cpu())

            val_accuracy = 100*num_correct / num_samples

            print(f"val_loss:{val_loss_epoch_avg}")
            print(f"Validation Accuracy: {val_accuracy:.2f}")

        if val_loss_epoch_avg < lowest_val_loss:
            best_epoch = epoch
            lowest_val_loss = val_loss_epoch_avg
            best_model_val_loss = model.state_dict()
            torch.save(best_model_val_loss, 'A/pretrained_weights_customCNN/best_model_val_loss_test_final.pth')
        elif epoch - best_epoch > early_stop_thresh:
            print(f"Early stopped training at epoch {epoch}" )
            print(f"Best Epoch is {best_epoch}")
            break  # terminate the training loop


    plt.figure(figsize=(10, 6))
    names = ["val_loss","train_loss"]
    plt.plot(val_loss_list, marker='o')
    plt.plot(train_loss_list, marker = 'o')
    plt.title('Model Performance Logging During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(names, loc='upper right')
    plt.grid(True)
    plt.savefig("loss image.png")
        #if val_accuracy > highest_val_accuracy:
        #    best_epoch = epoch
        #    highest_val_accuracy = val_accuracy
        #    best_model_val_acc = model.state_dict()
        #    torch.save(best_model_val_acc, 'A/pretrained_weights_customCNN/best_model_val_acc_test.pth')


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
    model.load_state_dict(torch.load("A/pretrained_weights_customCNN/best_model_val_loss_test_final.pth"))
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

        
        classes = ["0","1"]
        utils.evaluate_performance_metrics(true_labels, predictions, classes,"CNN_test")
        print(f"test_loss:{test_loss}")
        print(f"Test Accuracy: {num_correct / num_samples:.2f}")
        #print(f"roc auc score: {roc_auc}")

#train_customCNN(100,0.0001)
#test_customCNN()


