import torch
import argparse
import pickle
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from torchvision import transforms
import time
import threading

def one_hot_encode(labels, num_classes):
    return torch.nn.functional.one_hot(labels, num_classes=num_classes)

def load_data(data_path, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),  
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  
        transforms.RandomRotation(degrees=10),  
        transforms.GaussianBlur(kernel_size=3),  
        transforms.Resize((256, 256)),  
        transforms.ToTensor(),  
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   
    ])

    data_root = data_path
    
    print("Starting to load data...")

    # Create ImageFolder datasets for train, val, and test
    train_dataset = ImageFolder(root=data_root + '/train', transform=transform)
    val_dataset = ImageFolder(root=data_root + '/val', transform=transform)
    test_dataset = ImageFolder(root=data_root + '/test', transform=transform)

    # Create DataLoaders for train, val, and test
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Start a separate thread to display a spinning bar
    def show_spinner():
        spinner = ['-', '\\', '|', '/']
        idx = 0
        while True:
            print(f"\rData Loading in Process... {spinner[idx % len(spinner)]}", end="", flush=True)
            idx += 1
            time.sleep(0.1)

    spinner_thread = threading.Thread(target=show_spinner)
    spinner_thread.daemon = True
    spinner_thread.start()
    
    class_labels_dict = {class_name: label for label, class_name in enumerate(train_dataset.classes, start=0)}
    reverse_class_labels_dict = {label: class_name for class_name, label in class_labels_dict.items()}
    
    print("\n")
    for class_name in class_labels_dict:
        print(f"{class_name} has Class Index {class_labels_dict[class_name]}\n")

    for data, labels in train_loader:
        one_hot_labels = one_hot_encode(labels, num_classes=25)
            
    print("\nData loading completed.")
            
    return train_loader, val_loader, test_loader

def parse_arguments():
    parser = argparse.ArgumentParser(description='Data Loader Script')
    parser.add_argument('--data_path', type=str, required=True,
                        help='Path to the directory where data is located')
    
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size that must be used for loading data')
    
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers used for data loading')
    
    return parser.parse_args()

def save_data_loaders(train_loader, val_loader, test_loader, save_path):
    with open(save_path, 'wb') as f:
        pickle.dump((train_loader, val_loader, test_loader), f)
    print("Data loader saved successfully.\n")

if __name__ == "__main__":
    args = parse_arguments()
    data_path = args.data_path
    batch_size = args.batch_size
    num_workers = args.num_workers
    
    train_loader, val_loader, test_loader = load_data(data_path, batch_size, num_workers)
    save_data_loaders(train_loader, val_loader, test_loader, "data_loaders.pkl")
