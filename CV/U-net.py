import torch.nn as layers
import torch
import numpy as np  

class Doubleconv(layers.Module):
    def __init__(self,input_channels,output_channel):
        super().__init__()
        self.conv1 = layers.Conv2d(input_channels,output_channel,3,stride=1)
        self.norm1 = layers.BatchNorm2d(output_channel)
        self.conv2 = layers.Conv2d(output_channel,output_channel,3,stride=1)
        self.norm2 = layers.BatchNorm2d(output_channel)
        self.ReLU = layers.ReLU()
    def forward(self,x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.ReLU(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.ReLU(x)
        return x


class Model(layers.Module):
    def __init__(self,input_channels,output_channels):
        super().__init__()
        self.shapes = [input_channels,64, 128, 256,512,1024]
        self.doubleconv = layers.ModuleList([Doubleconv(self.shapes[i-1],self.shapes[i]) for i in range(1,len(self.shapes))])
        self.maxpol =  layers.MaxPool2d(2,stride=2)
    def forward(self,x):
        for i,l in enumerate(self.doubleconv):
            x = l(x)
            if i <= len(self.shapes)-1:
                x = self.maxpol(x)
        return x
    
def build_unet_model(input_channels=3, output_channels=1):
    # Your code here
    x = np.random.rand(1,input_channels,572,572)
    x = torch.Tensor(x)
    newmodel = Model(input_channels,output_channels)
    print(newmodel(x).shape)

build_unet_model()