import sys
import numpy as np
import io
import torch
import asyncio
from torch.optim import Adam
from torch.nn.functional import mse_loss
from torchvision import transforms
from PIL import Image

sys.path.insert(0, "src")
from src.model import VGG19Gatys
from src.utils import get_normalize_transform, denormalize, gram_matrix



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device.type)
model = VGG19Gatys().to(device).eval()

normalize = get_normalize_transform()
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
])

def load_image_tensor(image_bytes):
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    original_size = img.size
    tensor = preprocess(img)
    assert isinstance(tensor, torch.Tensor)
    tensor = tensor.unsqueeze(0).to(device)
    return tensor, original_size

async def run_style_transfer(content_bytes, style_bytes,alpha,websocket):

    content_tensor, original_size = load_image_tensor(content_bytes)
    style_tensor, _ = load_image_tensor(style_bytes)

    content_norm = normalize(content_tensor)
    style_norm = normalize(style_tensor)

    optimizing_img = content_norm.clone().requires_grad_(True)

    style_weights = [500.0, 300.0, 200.0, 100.0, 50.0]
    style_weights = [i * alpha for i in style_weights]


    with torch.no_grad():
        target_style_features, _ = model(style_norm)
        target_style_grams = [gram_matrix(f) for f in target_style_features]
        _, target_content_features = model(content_norm)

    base_content_weight = 1.0
    base_style_weight = 1e5 * max(0.01, alpha)

    optimizer = Adam([optimizing_img], lr=0.1*alpha)
    iter = 500
    for step in range(iter):
        optimizer.zero_grad()
        style_feats , content_feats = model(optimizing_img)

        content_loss = base_content_weight * mse_loss(content_feats.squeeze(0),
                       target_content_features.squeeze(0),reduction='mean')
        style_loss = 0.0

        for feat,target_gram,weight in zip(style_feats,target_style_grams,style_weights):
            gram = gram_matrix(feat)
            style_loss += weight *mse_loss(gram.squeeze(0),target_gram.squeeze(0),reduction='mean')

        style_loss *= base_style_weight
        loss = content_loss + style_loss
        loss.backward()
        optimizer.step()
        await websocket.send_text(str(step))
        if (step % 5) == 0 or step == iter-1:
            with torch.no_grad():
                output_tensor = denormalize(optimizing_img.clone())
                output_tensor = output_tensor.squeeze(0).cpu()

            output_np = (output_tensor.numpy() * 255).astype(np.uint8)
            output_np = np.transpose(output_np, (1,2,0))
            output_image = Image.fromarray(output_np).resize(original_size,Image.Resampling.LANCZOS)

            img_io = io.BytesIO()
            img_io = io.BytesIO()
            output_image.save(img_io, format='JPEG', quality=85)

            await websocket.send_bytes(img_io.getvalue())

            await asyncio.sleep(0)

