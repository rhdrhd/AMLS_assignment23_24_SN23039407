import utils
from monai.networks.nets import DenseNet121
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader

#load wandb
#wandb.login()
#wandb.init(project="aml-final-dense")

#select gpu training as priority
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The device used for training is", device)

#initialize the model
model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=9)

train_dataset, val_dataset, test_dataset = utils.load_dataset_t2()

def train_densenet121(num_epochs, lr):
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle= True, drop_last=True)

    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
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
            target = target.reshape(-1).cuda()
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
                target = target.reshape(-1).cuda()
                val_loss = criterion(val_output, target)
                _, pred_label = torch.max(val_output, 1)
                num_correct += (pred_label == target).sum().item()
                num_samples += pred_label.size(0)
                
            val_accuracy = 100*num_correct / num_samples
            print(f"val_loss:{val_loss}")
            print(f"Validation Accuracy: {val_accuracy:.2f}")

        if val_loss < lowest_val_loss:
            lowest_val_loss = val_loss
            best_model_val_loss = model.state_dict()
            torch.save(best_model_val_loss, 'B/pretrained_weights_denseNet/best_model_val_loss.pth')

        if val_accuracy > highest_val_accuracy:
            highest_val_accuracy = val_accuracy
            best_model_val_acc = model.state_dict()
            torch.save(best_model_val_acc, 'B/pretrained_weights_denseNet/best_model_val_acc.pth')

        #log performance on wandb platform
        #wandb.log({"epoch":epoch+1})
        #wandb.log({"train_loss":loss})
        #wandb.log({"val_accuracy": val_accuracy})
        #wandb.log({"val_loss":val_loss})

def test_denseNet121():
    test_loader = DataLoader(test_dataset, batch_size=64)
    model.load_state_dict(torch.load("B/pretrained_weights_denseNet/best_model_val_loss.pth"))
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    true_labels = []
    predictions = []

    with torch.no_grad():
        num_correct = 0
        num_samples = 0
        for data, target in test_loader:
            data = data.cuda()
            test_output = model(data)
            target = target.reshape(-1).long().cuda()
            test_loss = criterion(test_output, target)
            _, pred_label = torch.max(test_output, 1)
            predictions.extend(pred_label.cpu().numpy())
            true_labels.extend(target.cpu().numpy())
            num_correct += (pred_label == target).sum().item()
            num_samples += pred_label.size(0)

        
        classes = ["1","2","3","4","5","6","7","8","9"]
        #roc_auc = roc_auc_score(true_labels, predictions)
        utils.plot_confusion_matrix(true_labels, predictions, classes, 'denseNet121')
        print(f"test_loss:{test_loss}")
        print(f"Test Accuracy: {num_correct / num_samples:.2f}")
        #print(f"roc auc score: {roc_auc}")

test_denseNet121()