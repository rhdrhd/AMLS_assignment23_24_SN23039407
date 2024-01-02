from . import utils
from monai.networks.nets import DenseNet121
from torchvision.models import resnet18
import torch
import torch.nn as nn
import torch.optim as optim
import wandb
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

#load wandb
#wandb.login()
#wandb.init(project="aml-final-dense")

#select gpu training as priority
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("The device used for training is", device)

def select_model(model_name):
    if model_name == "DenseNet121":
        model = DenseNet121(spatial_dims=2, in_channels=3, out_channels=9)
    elif model_name == "ResNet18":
        model = resnet18()
    else:
        print("This model is not yet supported")
    return model 


def train_model(model_name, num_epochs, lr):
    #initialize the model
    model = select_model(model_name)
    
    train_dataset, val_dataset, test_dataset = utils.load_dataset_t2(model_name)

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
    train_loss_list = []
    val_loss_list = []
    best_epoch = 0
    early_stop_thresh = 10
    for epoch in range(num_epochs):
        total_val_loss_epoch = 0
        total_train_loss_epoch = 0
        model.train()
        for data, target in train_loader:

            optimizer.zero_grad()
            data = data.cuda()
            output = model(data)
            target = target.reshape(-1).cuda()
            loss = criterion(output, target)
            total_train_loss_epoch += loss
            loss.backward()
            optimizer.step()
        
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
                target = target.reshape(-1).cuda()
                val_loss = criterion(val_output, target)
                total_val_loss_epoch+= val_loss
                _, pred_label = torch.max(val_output, 1)
                num_correct += (pred_label == target).sum().item()
                num_samples += pred_label.size(0)
            
            val_loss_epoch_avg=  total_val_loss_epoch/len(val_loader)
            val_accuracy = 100*num_correct / num_samples

            val_loss_list.append(val_loss_epoch_avg.cpu())
            print(f"val_loss:{val_loss}")
            print(f"Validation Accuracy: {val_accuracy:.2f}")

        if val_loss_epoch_avg < lowest_val_loss:
            best_epoch = epoch
            lowest_val_loss = val_loss_epoch_avg
            best_model_val_loss = model.state_dict()
            torch.save(best_model_val_loss, f'B/pretrained_weights/best_model_val_loss_{model_name}.pth')
        elif epoch - best_epoch > early_stop_thresh:
            print(f"Early stopped training at epoch {epoch}" )
            print(f"Best Epoch is {best_epoch}")
            break  # terminate the training loop

        #if val_accuracy > highest_val_accuracy:
        #    highest_val_accuracy = val_accuracy
        #    best_model_val_acc = model.state_dict()
        #    torch.save(best_model_val_acc, 'B/pretrained_weights_denseNet/best_model_val_acc.pth')

    plt.figure(figsize=(10, 6))
    names = ["val_loss","train_loss"]
    plt.plot(val_loss_list, marker='o')
    plt.plot(train_loss_list, marker = 'o')
    plt.title('Model Performance Logging During Training')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(names, loc='upper right')
    plt.grid(True)
    plt.savefig(f"loss_image_{model_name}.png")        

        #log performance on wandb platform
        #wandb.log({"epoch":epoch+1})
        #wandb.log({"train_loss":loss})
        #wandb.log({"val_accuracy": val_accuracy})
        #wandb.log({"val_loss":val_loss})

def test_model(model_name):
    #initialize the model
    model = select_model(model_name)

    train_dataset, val_dataset, test_dataset = utils.load_dataset_t2(model_name) 

    test_loader = DataLoader(test_dataset, batch_size=64)
    model.load_state_dict(torch.load(f"B/pretrained_weights/best_model_val_loss_{model_name}.pth"))
    model.to(device)
    model.eval()

    criterion = nn.CrossEntropyLoss()

    true_labels = []
    predictions = []
    probabilities = []
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
            probabilities.extend(test_output.cpu().numpy())
            num_correct += (pred_label == target).sum().item()
            num_samples += pred_label.size(0)

        
        classes = ["1","2","3","4","5","6","7","8","9"]
        #roc_auc = roc_auc_score(true_labels, predictions)
        utils.evaluate_performance_metrics(true_labels, predictions,probabilities, classes, model_name)
        print(f"test_loss:{test_loss}")
        print(f"Test Accuracy: {num_correct / num_samples:.2f}")
        #print(f"roc auc score: {roc_auc}")

train_model("ResNet18",50,0.001)
test_model("ResNet18")