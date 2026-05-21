import os
import random
from PIL import Image
from torch.utils.data import Dataset


class NeuralArtTransferDataset(Dataset):
    def __init__(self, content_dir, style_dir, transform=None):
        self.content_paths = [
            os.path.join(content_dir, f)
            for f in os.listdir(content_dir)
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
        self.style_paths = []
        for root, dirs, files in os.walk(style_dir):
            for f in files:
                if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.style_paths.append(os.path.join(root, f))
        self.transform = transform
        print(f"--> Found {len(self.content_paths)} Content Images")
        print(f"--> Found {len(self.style_paths)} Style Images")

    def __len__(self):
        return len(self.content_paths)

    def __getitem__(self, idx):
        content_img = Image.open(self.content_paths[idx]).convert('RGB')
        style_idx = random.randint(0, len(self.style_paths) - 1)
        style_img = Image.open(self.style_paths[style_idx]).convert('RGB')
        if self.transform:
            content_img = self.transform(content_img)
            style_img = self.transform(style_img)
        return content_img, style_img