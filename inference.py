import io
import numpy as np
import onnxruntime as ort
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from PIL import Image

app = FastAPI(title="Neural Art Studio ONNX API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading ONNX Model into memory...")
ort_session = ort.InferenceSession("neural_art_transfer.onnx", providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])

def preprocess_image(image_bytes):
    """Converts raw HTTP binary files into the exact numpy arrays ONNX expects."""
    img = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    img = img.resize((512, 512), Image.Resampling.LANCZOS)
    
    img_array = np.array(img, dtype=np.float32) / 255.0
    
    img_array = np.transpose(img_array, (2, 0, 1))
    
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

@app.post("/api/style")
async def style_transfer_endpoint(content: UploadFile = File(...), style: UploadFile = File(...)):
    try:
        content_bytes = await content.read()
        style_bytes = await style.read()
        
        content_np = preprocess_image(content_bytes)
        style_np = preprocess_image(style_bytes)
        
        alpha_np = np.array([1.0], dtype=np.float32)
        
        inputs = {
            "content": content_np,
            "style": style_np,
            "alpha": alpha_np
        }
        
        outputs = ort_session.run(None, inputs)
        stylized_np =stylized_np = np.asarray(outputs[0])
        

        stylized_np = np.squeeze(stylized_np, axis=0)
        stylized_np = np.clip(stylized_np, 0.0, 1.0) * 255.0
        stylized_np = stylized_np.astype(np.uint8)
        
        stylized_np = np.transpose(stylized_np, (1, 2, 0))
        
        output_image = Image.fromarray(stylized_np)
        
        img_io = io.BytesIO()
        output_image.save(img_io, 'JPEG', quality=90)
        img_io.seek(0)
        
        return StreamingResponse(img_io, media_type="image/jpeg")
        
    except Exception as e:
        print(f"API Crash: {e}")
        return {"error": str(e)}, 500

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)