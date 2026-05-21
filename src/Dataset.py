import os
import random 
from PIL import Image
from torch.utils.data import Dataset,DataLoader
import torchvision.transforms as transforms

class NeuralArtTransferDataset(Dataset):
    def __init__(self,content_dir,style_dir,transform=None):
        self.content_paths = [os.path.join(content_dir,f) for f in os.listdir(content_dir) if f.endswith(('.jpg','.png','.jpeg'))]
        self.style_paths = [os.path.join(style_dir,f) for f in os.listdir(style_dir) if f.endswith(('.jpg','.png','.jpeg'))]
        self.transform = transform

    def __len__(self):

        return len(self.content_paths)
    
    def __getitem__(self,idx):
        content_path = self.content_paths[idx]
        style_path = random.choice(self.style_paths)

        content_image = Image.open(content_path).convert('RGB')
        style_image = Image.open(style_path).convert('RGB')

        if self.transform:
            content_image = self.transform(content_image)
            style_image = self.transform(style_image)

        return content_image,style_image