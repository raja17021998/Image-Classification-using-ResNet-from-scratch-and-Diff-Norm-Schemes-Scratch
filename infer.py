import argparse
import os
from PIL import Image
import torch
import torchvision.transforms as transforms
from resnet import ResNet, Block 
import numpy as np 


def inference(model_file, normalization, n, test_data_file, output_file):
    print("Performing inference...")
    print(f"Model file: {model_file}")
    print(f"Normalization scheme: {normalization}")
    print(f"Number of layers: {n}")
    print(f"Test data file: {test_data_file}")
    print(f"Output file: {output_file}")

    def ResNetModel(img_channel=3, num_classes=25, norm_type=normalization):
        return ResNet(Block, [n, n, n], img_channel, num_classes, norm_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = ResNetModel(img_channel=3, norm_type=normalization, num_classes=25).to(device)
    print(model.load_state_dict(torch.load(f'{model_file}', map_location=device)))
    model.eval()

    image_files = sorted(os.listdir(test_data_file))
    with open(output_file, 'w') as f:
        for image_file in image_files:
            image_path = os.path.join(test_data_file, image_file)
            image = Image.open(image_path)
            transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(),  
                    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  
                    transforms.RandomRotation(degrees=10),  
                    transforms.GaussianBlur(kernel_size=3),  
                    transforms.Resize((256, 256)),  
                    transforms.ToTensor(),  
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))   
                ])
            image = transform(image).unsqueeze(0).to(device)
            
             
             
            
            with torch.no_grad():
                output = model(image)
                predicted_class = torch.argmax(output, dim=1).item()
            f.write(f"{predicted_class}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Script for model inference")
    parser.add_argument("--model_file", type=str, help="Path to the trained model")
    parser.add_argument("--normalization", type=str, choices=["bn", "in", "bin", "ln", "gn", "nn", "inbuilt"],
                        help="Normalization scheme")
    parser.add_argument("--n", type=int, choices=[1, 2, 3], help="Number of layers")
    parser.add_argument("--test_data_file", type=str, help="Path to the directory containing the images")
    parser.add_argument("--output_file", type=str, help="File containing the prediction in the same order as the images in directory")

    args = parser.parse_args()

    if args.model_file is None or args.normalization is None or args.n is None or args.test_data_file is None or args.output_file is None:
        parser.error("Please provide all required arguments.")

    inference(args.model_file, args.normalization, args.n, args.test_data_file, args.output_file)
