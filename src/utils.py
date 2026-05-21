import torch
import torchvision.transforms as transforms

MEAN = torch.tensor([0.485, 0.456, 0.406])
STD = torch.tensor([0.229, 0.224, 0.225])

def get_normalize_transform():
    return transforms.Normalize(mean=MEAN.tolist(), std=STD.tolist())

def denormalize(tensor):
    device = tensor.device
    mean = MEAN.view(1, 3, 1, 1).to(device)
    std = STD.view(1, 3, 1, 1).to(device)
    tensor = tensor * std + mean
    return tensor.clamp(0, 1)

def gram_matrix(tensor):
    b, c, h, w = tensor.size()
    features = tensor.view(b, c, h * w)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (c * h * w)
    return gram

def total_variation_loss(img):
    tv_h = torch.sum(torch.abs(img[:, :, 1:, :] - img[:, :, :-1, :]))
    tv_w = torch.sum(torch.abs(img[:, :, :, 1:] - img[:, :, :, :-1]))
    return tv_h + tv_w