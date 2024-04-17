import torch.nn as nn 
from norm import StandardBatchNorm, InstanceNorm, BatchInstanceNorm, LayerNorm, GroupNorm, IdentityNorm


class Block(nn.Module):
    def __init__(self, in_channels, intermediate_channels, identity_downsample=None, stride=1, norm_type='bn'):
        super().__init__()
        self.expansion = 2
        self.conv1 = nn.Conv2d(in_channels, intermediate_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        
        if norm_type == 'bn':
            self.nl1 = StandardBatchNorm(intermediate_channels)
            self.nl2 = StandardBatchNorm(intermediate_channels* self.expansion)
        elif norm_type == 'in':
            self.nl1 = InstanceNorm(intermediate_channels)
            self.nl2 = InstanceNorm(intermediate_channels* self.expansion)
        elif norm_type == 'gn':
            self.nl1 = GroupNorm(num_features=intermediate_channels)  # Assuming a group size of 4
            self.nl2 = GroupNorm(num_features=intermediate_channels* self.expansion)
        elif norm_type == 'bin':
            self.nl1 = BatchInstanceNorm(intermediate_channels)
            self.nl2 = BatchInstanceNorm(intermediate_channels* self.expansion)
        elif norm_type == 'ln':
            self.nl1 = LayerNorm(num_features=intermediate_channels)
            self.nl2 = LayerNorm(num_features=intermediate_channels* self.expansion)
        elif norm_type=='nn':
            self.nl1= IdentityNorm(intermediate_channels)
            self.nl2 = IdentityNorm(intermediate_channels* self.expansion)
            
        self.conv2 = nn.Conv2d(intermediate_channels, intermediate_channels * self.expansion, kernel_size=3, stride=1, padding=1, bias=False)    
        self.relu = nn.ReLU()
        self.identity_downsample = identity_downsample
        self.stride = stride

    def forward(self, x):
        identity = x.clone()

        x = self.conv1(x)
        x = self.nl1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.nl2(x)

        if self.identity_downsample is not None:
            identity = self.identity_downsample(identity)

        x += identity
        x = self.relu(x)
        return x

class ResNet(nn.Module):
    def __init__(self, block, layers, image_channels, num_classes, norm_type):
        super(ResNet, self).__init__()
        self.in_channels = 16
        self.conv1 = nn.Conv2d(image_channels, 16, kernel_size=3, stride=1, padding=1, bias=False)

        if norm_type == 'bn':
            self.norm = StandardBatchNorm(16)
            self.gamma = self.norm.gamma
            self.beta = self.norm.beta
            
        elif norm_type == 'in':
            self.norm = InstanceNorm(16)
            self.gamma = self.norm.gamma
            self.beta = self.norm.beta
            
        elif norm_type == 'gn':
            self.norm = GroupNorm(group=4, num_features=16)  # Assuming a group size of 4
            self.gamma = self.norm.gamma
            self.beta = self.norm.beta
            
        elif norm_type == 'bin':
            self.norm = BatchInstanceNorm(16)
            self.gamma = self.norm.gamma
            self.beta = self.norm.beta
            
        elif norm_type == 'ln':
            self.norm = LayerNorm(16)
            self.gamma = self.norm.gamma
            self.beta = self.norm.beta
            
        elif norm_type=='nn':
            self.norm= IdentityNorm(16)
            self.gamma = self.norm.gamma
            self.beta = self.norm.beta
        else:
            raise ValueError("Invalid normalization type. Choose from 'batch', 'instance', or 'group'.")
        self.relu = nn.ReLU()
        
        self.layer1 = self._make_layer(block, layers[0], intermediate_channels=16, stride=1, norm_type=norm_type)
        self.layer2 = self._make_layer(block, layers[1], intermediate_channels=32, stride=2, norm_type=norm_type)
        self.layer3 = self._make_layer(block, layers[2], intermediate_channels=64, stride=2, norm_type=norm_type)
        
        self.avgpool = nn.AdaptiveAvgPool2d((8,8))
        self.fc = nn.Linear(128 * 8*8, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x
    
    
    def _make_layer(self, block, num_residual_blocks, intermediate_channels, stride, norm_type):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != intermediate_channels * 2:
            if norm_type == 'gn':
                num_groups = 4  # Adjust according to your group size preference
                identity_downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, intermediate_channels * 2, kernel_size=1, stride=stride, bias=False),
                    GroupNorm(num_features=intermediate_channels * 2, group=num_groups),
                )
            elif norm_type == 'bn':
                identity_downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, intermediate_channels * 2, kernel_size=1, stride=stride, bias=False),
                    StandardBatchNorm(intermediate_channels * 2)
                )
            elif norm_type == 'in':
                identity_downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, intermediate_channels * 2, kernel_size=1, stride=stride, bias=False),
                    InstanceNorm(intermediate_channels * 2)
                )
            elif norm_type == 'bin':
                identity_downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, intermediate_channels * 2, kernel_size=1, stride=stride, bias=False),
                    BatchInstanceNorm(intermediate_channels * 2)
                )
            elif norm_type == 'ln':
                identity_downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, intermediate_channels * 2, kernel_size=1, stride=stride, bias=False),
                    LayerNorm(intermediate_channels * 2)
                )
            elif norm_type == 'nn':
                identity_downsample = nn.Sequential(
                    nn.Conv2d(self.in_channels, intermediate_channels * 2, kernel_size=1, stride=stride, bias=False),
                    IdentityNorm(intermediate_channels * 2)
                )

        layers.append(block(self.in_channels, intermediate_channels, identity_downsample, stride, norm_type))
        self.in_channels = intermediate_channels * 2

        for i in range(num_residual_blocks - 1):
            layers.append(block(self.in_channels, intermediate_channels, norm_type=norm_type))

        return nn.Sequential(*layers)
