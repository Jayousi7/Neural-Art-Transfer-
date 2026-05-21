import torch
import torch.nn as nn
import torchvision.models as models
from src.utils import adain

class ResNet50Encoder(nn.Module):
    def __init__(self):

        super(ResNet50Encoder, self).__init__()

        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1) 
        resnet_layers = list(resnet.children())
    
        self.stage1= nn.Sequential(*resnet_layers[:4])

        self.stage2 = resnet_layers[4]
        self.stage3 = resnet_layers[5]
        self.stage4 = resnet_layers[6]

        
        for param in self.parameters():
            param.requires_grad = False
        
    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)

        return x

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        #input shape from RestNet50 layer3 is (batch,1024,H/16,W/16)

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),  
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2,mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256,64,kernel_size=3,stride=1,padding=1),
            nn.ReLU(),

            nn.Upsample(scale_factor=2,mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64,3,kernel_size=3,stride=1,padding=1),
        )
    def forward(self,x):
        return self.decoder(x)


class StyleTransferModel(nn.Module):
    def __init__(self,encoder,decoder):
        super(StyleTransferModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self,content_img,style_img,alpha=1.0):
        content_feat = self.encoder(content_img)
        style_feat = self.encoder(style_img)

        blended_feat = adain(content_feat,style_feat,alpha)

        stylized_img = self.decoder(blended_feat)

        return stylized_img