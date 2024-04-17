import torch 
import torch.nn as nn 


class GroupNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, group=4, affine=True):
        super(GroupNorm, self).__init__()
        self.eps = eps
        self.num_features = num_features
        self.affine = affine
        self.group = group

        if self.affine:
            self.scale = nn.Parameter(torch.ones(1, num_features, 1, 1))
            self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))

    def _initialize_parameters(self):
        if self.affine:
            nn.init.ones_(self.scale)
            nn.init.zeros_(self.bias)

    def forward(self, x):
        N, C, H, W = x.shape
        # input_tensor = torch.randn(torch.Size([4, 3, 224, 224]))
        assert C % self.group == 0
        assert self.num_features == C

        x = x.view(N, self.group, C // self.group, H, W)
        axes = (1, 2, 3)
        mean = x.mean(dim=axes, keepdim=True)
        variance = x.var(dim=axes, keepdim=True)
        denominator = torch.sqrt(variance + self.eps)
        x = (x - mean) / denominator
        x = x.view(N, C, H, W)

        if self.affine:
            x = x * self.scale + self.bias

        return x
    
# Create an instance of BatchNorm module
num_features = 4
batch_norm = GroupNorm(num_features)

# Set the module to evaluation mode
batch_norm.eval()

# Generate a random input tensor
batch_size = 4
num_channels = num_features
height = 16
width = 16
input_tensor = torch.randn(torch.Size([4, 3, 224, 224]))



# Normalize the input tensor using BatchNorm module
output_tensor = batch_norm.forward(input_tensor)

# Print the shapes of input and output tensors
print("Shape of input tensor:", input_tensor.shape)
# print(f"\nInput tesnor is: {input_tensor}\n")
print("Shape of output tensor:", output_tensor.shape)
# print(f"\nInput tesnor is: {output_tensor}\n")