import argparse
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image
import torch
from torchvision.transforms import Compose, Resize, ToTensor
from PIL import Image
import numpy as np
import cv2
from matplotlib import pyplot as plt
from resnet import ResNet, Block


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def visualize_cam(model_name, image_path,n, batch_size,norm_type, target_class):
    folder_name = f'diff_models/{model_name}'
    
    
    def ResNetModel(img_channel=3, num_classes=25, norm_type= norm_type):
        print(f"\nNorm in Gradcam is: {norm_type}\n")
        return ResNet(Block, [n,n,n], img_channel, num_classes, norm_type)
    
    
    
    model = ResNetModel(img_channel=3, norm_type=norm_type, num_classes=25).to(device)   
            
            
    model.load_state_dict(torch.load(f'{folder_name}/{model_name}_model.pth', map_location=device))
    

    parameters = list(model.parameters())
    
    print(parameters)
    
    model.eval()
    
    target_layers = [model.layer3[-1]]
    cam = GradCAM(model=model, target_layers=target_layers)
    
    targets = [ClassifierOutputTarget(target_class)]

    transform = Compose([
        Resize((256, 256)),
        ToTensor()
    ])

    image = Image.open(image_path)
    image_tensor = transform(image).unsqueeze(0)

    rgb_img = cv2.imread(image_path, 1)[:, :, ::-1]
    rgb_img = np.float32(rgb_img) / 255
    grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0, :]
    visualization = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    model_outputs = cam.outputs
    print(f"Model outputs: {model_outputs}")
    plt.imshow(visualization)
    plt.show()
    
def parse_arguments():
    parser = argparse.ArgumentParser(description='Grad Cam Script')
    
    parser.add_argument('--model_name', type=str,
                        help="details of model hyper params")
    
    parser.add_argument('--n', type=int, default=2,
                        help='Number of layers in ResNet')
    
    
    parser.add_argument('--target_class', type=int, default=0,
                        help='Target Class For Visualising')
    
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch Size')
    
    
    parser.add_argument('--norm_type', type=str, default="bn",
                        help='Normalization to be used: [BN, IN, BIN, LN, GN, and NN] to be allowed')
    
    parser.add_argument('--image_path', type=str, 
                        help='Path to the input image')
    

    
    
    

    return parser.parse_args()

if __name__ == "__main__":
    
    args = parse_arguments()
    model_name= args.model_name 
    n= args.n 
    batch_size = args.batch_size
    norm_type = args.norm_type
    image_path= args.image_path 
    target_class= args.target_class 
    
    visualize_cam(model_name, image_path,n, batch_size,norm_type, target_class)
