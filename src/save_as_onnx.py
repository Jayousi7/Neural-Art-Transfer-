import onnx 
import onnxruntime 
import torch
from model import ResNet50Encoder, Decoder, StyleTransferModel 

def export_to_onnx():
    print("Loading architectures")
    encoder = ResNet50Encoder()
    decoder = Decoder()
    model = StyleTransferModel(encoder, decoder)

    weights_path = "checkpoints/decoder_epoch_5.pth" 
    
    print(f"Loading weights from {weights_path}...")
    decoder.load_state_dict(torch.load(weights_path, map_location="cpu"))
    
    model.eval()
    model.to("cpu")


    dummy_content = torch.randn(1, 3, 512, 512)
    dummy_style = torch.randn(1, 3, 512, 512)
    
    dummy_alpha = torch.tensor([1.0], dtype=torch.float32)

    onnx_file_path = "neural_art_transfer.onnx"

    print("Tracing the computational graph")
    torch.onnx.export(
        model, 
        args=(dummy_content, dummy_style, dummy_alpha),
        f=onnx_file_path,
        export_params=True,
        opset_version=14,         
        do_constant_folding=True, 
        input_names=["content", "style", "alpha"],
        output_names=["stylized_output"],
        
        dynamic_axes={
            "content": {0: "batch_size", 2: "height", 3: "width"},
            "style": {0: "batch_size", 2: "height", 3: "width"},
            "stylized_output": {0: "batch_size", 2: "height", 3: "width"}
        }
    )
    
    print(f"Export successful! Production model saved to: {onnx_file_path}")

if __name__ == "__main__":
    export_to_onnx()