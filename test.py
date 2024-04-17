# import time 
# import torch 
# import torch.nn as nn 
# import torch.optim as optim 
# from tqdm import tqdm 
# from sklearn.metrics import f1_score, accuracy_score
# import numpy as np
# import matplotlib.pyplot as plt 
# from resnet import ResNet, Block 
# import argparse
# import pickle


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# def test(model_name,n,test_loader, norm_type):
    
#     folder_name = f'{model_name}'
    
#     def ResNetModel(img_channel=3, num_classes=25, norm_type= norm_type):
#         print(f"\nNorm in Test is: {norm_type}\n")
#         return ResNet(Block, [n,n,n], img_channel, num_classes, norm_type)
    
    
    
#     model = ResNetModel(img_channel=3, norm_type=norm_type, num_classes=25).to(device)   
            
            
#     model.load_state_dict(torch.load(f'{folder_name}/{model_name}_model.pth', map_location=device))

#     model.eval()  # Set the model to evaluation mode

#     all_test_labels = []
#     all_predictions = []

#     with torch.no_grad():
#         for test_data, test_labels in tqdm(test_loader, desc='Inference on Test Set', leave=False):
#             test_x = test_data.to(device)
#             test_y = test_labels.to(device)

#             # Forward pass
#             test_y_hat = model(test_x)

#             # Record predictions for comparison
#             all_test_labels.extend(test_y.cpu().numpy())
#             all_predictions.extend(torch.argmax(test_y_hat, dim=1).cpu().numpy())

#     # Convert the labels and predictions to numpy arrays
#     all_test_labels = np.array(all_test_labels)
#     all_predictions = np.array(all_predictions)
            
#     return all_test_labels, all_predictions
        
        
# def load_saved_data_loaders(file_path):
#     with open(file_path, 'rb') as f:
#         train_loader, val_loader, test_loader = pickle.load(f)
#     return test_loader

# def parse_arguments():
#     parser = argparse.ArgumentParser(description='Testing Script')
    
#     parser.add_argument('--model_name', type=str,
#                         help="details of model hyper params")
    
#     parser.add_argument('--n', type=int, default=2,
#                         help='Number of layers in ResNet')
    
#     parser.add_argument('--batch_size', type=int, default=32,
#                         help='Batch Size')
    
    
#     parser.add_argument('--norm_type', type=str, default="bn",
#                         help='Normalization to be used: [BN, IN, BIN, LN, GN, and NN] to be allowed')
    

    
    
    

#     return parser.parse_args()
        
# def main():
#     args = parse_arguments()
#     model_name= args.model_name 
#     batch_size = args.batch_size
#     norm_type = args.norm_type
#     n = args.n

    
#     test_loader = load_saved_data_loaders("data_loaders.pkl")
#     all_test_labels, all_predictions= test(model_name,n,test_loader, norm_type)
    
#     ctr=0 
    
#     for label,pred in zip(all_test_labels, all_predictions):
#         ctr+=1
#         if ctr% 100==0:
#             print(f"For label: {label}, prediction is {pred}")
        
# if __name__ == "__main__":
#     main()

import torch
import torch.nn as nn
from resnet import ResNet, Block  # Assuming 'resnet.py' contains your ResNet implementation
import argparse
import pickle
from PIL import Image
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def test_single_image(model_name, n, image_path, norm_type):
    def ResNetModel(img_channel=3, num_classes=25, norm_type=norm_type):
        print(f"\nNorm in Test is: {norm_type}\n")
        return ResNet(Block, [n, n, n], img_channel, num_classes, norm_type)

    model = ResNetModel(img_channel=3, norm_type=norm_type, num_classes=25).to(device)
    model.load_state_dict(torch.load(f'{model_name}/{model_name}_model.pth', map_location=device))
    print(model, end="\n")
    model.eval()  # Set the model to evaluation mode
    
    

    # Load and preprocess the image
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    image = transform(image).unsqueeze(0).to(device)  # Add batch dimension

    with torch.no_grad():
        output = model(image)
        print(f"output: {output}\n")
        predicted_class = torch.argmax(output, dim=1).item()
        
    return predicted_class

def parse_arguments():
    parser = argparse.ArgumentParser(description='Testing Script')
    
    parser.add_argument('--model_name', type=str, help="details of model hyper params")
    parser.add_argument('--image_path', type=str, help='Path to the input image')
    parser.add_argument('--n', type=int, default=2, help='Number of layers in ResNet')
    parser.add_argument('--norm_type', type=str, default="bn", help='Normalization to be used: [BN, IN, BIN, LN, GN, and NN] to be allowed')
    
    return parser.parse_args()

def main():
    args = parse_arguments()
    model_name = args.model_name
    image_path = args.image_path
    norm_type = args.norm_type
    n = args.n

    predicted_class = test_single_image(model_name, n, image_path, norm_type)
    print(f"Predicted class: {predicted_class}")

if __name__ == "__main__":
    main()
