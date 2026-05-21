import io
import sys
import numpy as np
import torch
import torch.nn as nn
from torch.optim import LBFGS
from torchvision import transforms
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

sys.path.insert(0, "src")
from model import VGG19Gatys
from utils import get_normalize_transform, denormalize, gram_matrix

app = FastAPI(title="Neural Art Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

@app.post("/api/style")
async def style_transfer_endpoint(
    content: UploadFile = File(...),
    style: UploadFile = File(...),
    alpha: float = Form(1.0),
):
    try:
        content_bytes = await content.read()
        style_bytes = await style.read()

        content_tensor, original_size = load_image_tensor(content_bytes)
        style_tensor, _ = load_image_tensor(style_bytes)

        content_norm = normalize(content_tensor)
        style_norm = normalize(style_tensor)

        optimizing_img = content_norm.clone().requires_grad_(True)

        style_weights = [1.0, 1.0, 1.0, 1.0, 1.0] 
        
        with torch.no_grad():
            target_style_features, _ = model(style_norm)
            target_style_grams = [gram_matrix(f) for f in target_style_features]
            _, target_content_features = model(content_norm)

        base_content_weight = 1.0
        base_style_weight = 1000000.0 * max(0.01, alpha) 
        
        optimizer = LBFGS([optimizing_img], max_iter=100)

        run = [0]
        def closure():
            optimizer.zero_grad()
            
            style_feats, content_feat = model(optimizing_img)
            
            content_loss = base_content_weight * nn.functional.mse_loss(content_feat.squeeze(0), target_content_features.squeeze(0), reduction='mean')
            
            style_loss = 0.0
            for feat, target_gram, weight in zip(style_feats, target_style_grams, style_weights):
                gram = gram_matrix(feat)
                style_loss += weight * nn.functional.mse_loss(gram.squeeze(0), target_gram.squeeze(0), reduction='mean')
            
            style_loss *= base_style_weight

            loss = content_loss + style_loss
            loss.backward()

            run[0] += 1
            return loss

        optimizer.step(closure)

        with torch.no_grad():
            output_tensor = denormalize(optimizing_img)
            output_tensor = output_tensor.squeeze(0).cpu()
            
        output_np = (output_tensor.numpy() * 255.0).astype(np.uint8)
        output_np = np.transpose(output_np, (1, 2, 0))

        output_image = Image.fromarray(output_np)
        output_image = output_image.resize(original_size, Image.Resampling.LANCZOS)

        img_io = io.BytesIO()
        output_image.save(img_io, "JPEG", quality=90)
        img_io.seek(0)

        return StreamingResponse(img_io, media_type="image/jpeg")

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)