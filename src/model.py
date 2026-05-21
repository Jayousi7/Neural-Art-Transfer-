import torch
import torch.nn as nn
import torchvision.models as models

class VGG19Gatys(nn.Module):
    def __init__(self):
        super(VGG19Gatys, self).__init__()
        vgg = models.vgg19(weights=models.VGG19_Weights.DEFAULT).features
        
        self.slice1 = nn.Sequential(*list(vgg.children())[:2])
        self.slice2 = nn.Sequential(*list(vgg.children())[2:7])
        self.slice3 = nn.Sequential(*list(vgg.children())[7:12])
        self.slice4 = nn.Sequential(*list(vgg.children())[12:21])
        self.slice5 = nn.Sequential(*list(vgg.children())[21:22])
        self.slice6 = nn.Sequential(*list(vgg.children())[22:30])
        
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        h_relu1_1 = self.slice1(x)
        h_relu2_1 = self.slice2(h_relu1_1)
        h_relu3_1 = self.slice3(h_relu2_1)
        h_relu4_1 = self.slice4(h_relu3_1)
        h_conv4_2 = self.slice5(h_relu4_1)
        h_relu5_1 = self.slice6(h_conv4_2)
        
        style_features = [h_relu1_1, h_relu2_1, h_relu3_1, h_relu4_1, h_relu5_1]
        content_features = h_conv4_2
        
        return style_features, content_features