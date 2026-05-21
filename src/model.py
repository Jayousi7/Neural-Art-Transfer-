import torch
import torch.nn as nn
import torchvision.models as models
from utils import adain

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
        h1 = self.stage1(x)
        h2= self.stage2(h1)
        h3= self.stage3(h2)
        h4 = self.stage4(h3)

        return h1, h2, h3, h4

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        #input shape from RestNet50 layer3 is (batch,1024,H/16,W/16)

        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),  
            nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),

            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),

            nn.Upsample(scale_factor=2,mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(256,64,kernel_size=3,stride=1,padding=0),
            nn.ReLU(),

            nn.Upsample(scale_factor=2,mode='nearest'),
            nn.ReflectionPad2d(1),
            nn.Conv2d(64,3,kernel_size=3,stride=1,padding=0),
        )
    def forward(self,x):
        return self.decoder(x)


class StyleTransferModel(nn.Module):
    def __init__(self,encoder,decoder):
        super(StyleTransferModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    
    def forward(self,content_img,style_img,alpha=1.0):
        content_feats = self.encoder(content_img)
        style_feats = self.encoder(style_img)

        # Grab only the deepest layer for the AdaIN transfer
        blended_feat = adain(content_feats[-1], style_feats[-1], alpha)

        stylized_img = self.decoder(blended_feat)

        return stylized_img