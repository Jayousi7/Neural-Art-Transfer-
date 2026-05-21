import argparse
import onnx
import onnxruntime
import torch
from model import ResNet50Encoder, Decoder, StyleTransferModel


def export_to_onnx(checkpoint_path, output_path="neural_art_transfer.onnx"):
    print("Loading architectures")
    encoder = ResNet50Encoder()
    decoder = Decoder()
    model = StyleTransferModel(encoder, decoder)

    print(f"Loading weights from {checkpoint_path}...")
    decoder.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))

    model.eval()
    model.to("cpu")

    dummy_content = torch.randn(1, 3, 512, 512)
    dummy_style = torch.randn(1, 3, 512, 512)
    dummy_alpha = torch.tensor([1.0], dtype=torch.float32)

    print("Tracing the computational graph")
    torch.onnx.export(
        model,
        args=(dummy_content, dummy_style, dummy_alpha),
        f=output_path,
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
    print(f"Export successful! Production model saved to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="neural_art_transfer.onnx")
    args = parser.parse_args()
    export_to_onnx(args.checkpoint, args.output)