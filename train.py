import time 
import torch 
import torch.nn as nn 
import torch.optim as optim 
from tqdm import tqdm 
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt 
from resnet import ResNet, Block 
import argparse
import pickle
import os 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model_name, n, batch_size, num_epochs, use_early_stopping, patience, num_classes, opt, lr,norm_type, num_workers,train_loader, val_loader, test_loader):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)



    num_epochs = num_epochs
    num_classes= num_classes

    def ResNetModel(img_channel=3, num_classes=25, norm_type=norm_type):
        print(f"\nNorm in Train is: {norm_type}\n")
        return ResNet(Block, [n,n,n], img_channel, num_classes, norm_type)


    model = ResNetModel(img_channel=3, norm_type= norm_type,num_classes=25).to(device)
    criterion = nn.CrossEntropyLoss()
    if opt=="SGD":
        optimizer = optim.SGD(model.parameters(), lr=lr)
        
    if opt=="AdaGrad":
        optimizer = optim.Adagrad(model.parameters(), lr=lr)
        
    if opt=="RMSprop":
        optimizer = optim.RMSprop(model.parameters(), lr=lr)
        
    if opt=="Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr)
        
        
    
    model= model.to(device)
    # Set the number of epochs
    train_losses = []
    train_accuracies = []
    train_micro_f1_scores = []
    train_macro_f1_scores = []
    val_losses = []
    val_accuracies = []
    val_micro_f1_scores = []
    val_macro_f1_scores = []

    # Early stopping parameters
    use_early_stopping = use_early_stopping
    patience = patience 
    early_stopping_counter = 0
    best_val_loss = np.inf
    
    folder_name = f'model_name_{model_name}'
    
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        
    csv_file_path = os.path.join(folder_name, f'{model_name}_metrics.csv')
    
    with open(csv_file_path, 'w') as f:
        f.write("Epoch,Train Loss,Train Accuracy,Train Micro F1,Train Macro F1,Val Loss,Val Accuracy,Val Micro F1,Val Macro F1\n")
        
    

    # Training loop
    total_start_time = time.time()
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        model.train()
        running_loss = 0
        all_labels = []
        all_predictions = []

        for data, labels in tqdm(train_loader, desc=f'Training - Epoch {epoch + 1}/{num_epochs}', leave=False):
            x = data.to(device)
            y = labels.to(device)

            optimizer.zero_grad()
            y_hat = model(x)
            loss = criterion(y_hat, y)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            all_labels.extend(y.cpu().numpy())
            all_predictions.extend(torch.argmax(y_hat, dim=1).cpu().numpy())

        epoch_train_time = time.time() - epoch_start_time
        average_loss = running_loss / len(train_loader)
        accuracy = accuracy_score(all_labels, all_predictions)
        micro_f1 = f1_score(all_labels, all_predictions, average='micro')
        macro_f1 = f1_score(all_labels, all_predictions, average='macro')

        print(f"Train - Epoch [{epoch + 1}/{num_epochs}], Loss: {average_loss:.4f}, Accuracy: {accuracy:.4f}, Micro F1: {micro_f1:.4f}, Macro F1: {macro_f1:.4f}, Time: {epoch_train_time:.2f} seconds")

        train_losses.append(average_loss)
        train_accuracies.append(accuracy)
        train_micro_f1_scores.append(micro_f1)
        train_macro_f1_scores.append(macro_f1)

        epoch_val_start_time = time.time()
        model.eval()
        val_running_loss = 0
        val_all_labels = []
        val_all_predictions = []

        with torch.no_grad():
            for val_data, val_labels in tqdm(val_loader, desc=f'Validation - Epoch {epoch + 1}/{num_epochs}', leave=False):
                val_x = val_data.to(device)
                val_y = val_labels.to(device)
                val_y_hat = model(val_x)
                val_loss = criterion(val_y_hat, val_y)
                val_running_loss += val_loss.item()
                val_all_labels.extend(val_y.cpu().numpy())
                val_all_predictions.extend(torch.argmax(val_y_hat, dim=1).cpu().numpy())

        epoch_val_time = time.time() - epoch_val_start_time
        val_average_loss = val_running_loss / len(val_loader)
        val_accuracy = accuracy_score(val_all_labels, val_all_predictions)
        val_micro_f1 = f1_score(val_all_labels, val_all_predictions, average='micro')
        val_macro_f1 = f1_score(val_all_labels, val_all_predictions, average='macro')

        print(f"Validation - Epoch [{epoch + 1}/{num_epochs}], Loss: {val_average_loss:.4f}, Accuracy: {val_accuracy:.4f}, Micro F1: {val_micro_f1:.4f}, Macro F1: {val_macro_f1:.4f}, Time: {epoch_val_time:.2f} seconds")

        val_losses.append(val_average_loss)
        val_accuracies.append(val_accuracy)
        val_micro_f1_scores.append(val_micro_f1)
        val_macro_f1_scores.append(val_macro_f1)

        if use_early_stopping:
            if val_average_loss < best_val_loss:
                best_val_loss = val_average_loss
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1

            if early_stopping_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs without improvement.")
                break
        
        with open(csv_file_path, 'a') as f:
            f.write(f"{epoch + 1},{average_loss},{accuracy},{micro_f1},{macro_f1},{val_average_loss},{val_accuracy},{val_micro_f1},{val_macro_f1}\n")
            
            
        print(f"File created and saved successfully!!\n")
            
        torch.save(model.state_dict(), f'{folder_name}/{model_name}_model.pth')
        print("Saved Model !!")

    total_train_time = time.time() - total_start_time
    print(f"Total Training Time: {total_train_time / 60:.2f} minutes")





    def plot_with_grid(x, train_data, val_data, train_label, val_label, xlabel, ylabel, title, x_interval=5, folder_name=None):
        plt.figure(figsize=(8, 4))
        plt.plot(x, train_data, label=train_label)
        plt.plot(x, val_data, label=val_label)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend()
        plt.xticks(np.arange(min(x), max(x)+1, x_interval))
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        if folder_name is not None:
            os.makedirs(folder_name, exist_ok=True)
            file_path = os.path.join(folder_name, f"{title.replace(' ', '_').lower()}.png")
            plt.savefig(file_path)
        plt.show()

    # Training Loss
    plot_with_grid(range(len(train_losses)), train_losses, val_losses, 'Train', 'Validation', 'Epoch', 'Loss', 'Training and Validation Loss Curves', folder_name=folder_name)

    # Training Accuracy
    plot_with_grid(range(len(train_accuracies)), train_accuracies, val_accuracies, 'Train', 'Validation', 'Epoch', 'Accuracy', 'Training and Validation Accuracy Curves', folder_name=folder_name)

    # Micro F1 Scores
    plot_with_grid(range(len(train_micro_f1_scores)), train_micro_f1_scores, val_micro_f1_scores, 'Train Micro F1', 'Validation Micro F1', 'Epoch', 'F1 Score', 'Micro F1 Score Curves', folder_name=folder_name)

    # Macro F1 Scores
    plot_with_grid(range(len(train_macro_f1_scores)), train_macro_f1_scores, val_macro_f1_scores, 'Train Macro F1', 'Validation Macro F1', 'Epoch', 'F1 Score', 'Macro F1 Score Curves', folder_name=folder_name)
    
    return train_losses, train_accuracies, train_micro_f1_scores,train_macro_f1_scores, val_losses, val_accuracies,val_micro_f1_scores, val_macro_f1_scores


def parse_arguments():
    parser = argparse.ArgumentParser(description='Training Script')
    
    parser.add_argument('--model_name', type=str,
                        help="details of model hyper params")
    
    parser.add_argument('--n', type=int, default=2,
                        help='Number of layers in ResNet')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch Size')
    
    parser.add_argument('--num_epochs', type=int, default=2,
                        help='Number of epochs to be trained for')
    
    parser.add_argument('--early_stopping', type=bool, default=False,
                        help='Boolean to indicate useage of Early Stopping')
    
    parser.add_argument('--patience', type=int, default=3,
                        help='How many epochs to wait')
    
    parser.add_argument('--num_classes', type=int, default=25,
                        help='Number of Classes in Dataset')
    
    parser.add_argument('--opt', type=str, default="Adam",
                        help='Optimizer to be used [SGD, AdaGrad, RMSprop, Adam]')
    
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate to be used in optimizer')
    
    parser.add_argument('--norm_type', type=str, default="bn",
                        help='Normalization to be used: [BN, IN, BIN, LN, GN, and NN] to be allowed')
    
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers used for data loading')
    
    return parser.parse_args()


def load_saved_data_loaders(file_path):
    with open(file_path, 'rb') as f:
        train_loader, val_loader, test_loader = pickle.load(f)
    return train_loader, val_loader, test_loader


def main():
    args = parse_arguments()
    model_name= args.model_name 
    n = args.n
    batch_size = args.batch_size
    num_epochs = args.num_epochs
    num_classes = args.num_classes
    opt = args.opt
    lr = args.lr
    norm_type = args.norm_type
    num_workers = args.num_workers
    use_early_stopping= args.early_stopping 
    patience= args.patience 
    
    train_loader, val_loader, test_loader = load_saved_data_loaders("data_loaders.pkl")
    train_losses, train_accuracies, train_micro_f1_scores,train_macro_f1_scores, val_losses, val_accuracies,val_micro_f1_scores, val_macro_f1_scores= train(model_name, n, batch_size, num_epochs,use_early_stopping,patience,num_classes, opt, lr,norm_type, num_workers,train_loader, val_loader, test_loader)
    

if __name__ == "__main__":
    main()
    
