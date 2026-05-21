# Neural Art Studio

A neural style transfer app that blends the content of one image with the artistic style of another. Built with PyTorch (ResNet50 encoder + custom decoder), served as a web app via FastAPI + ONNX Runtime.

![preview](static/preview.png)

## Quick Start (Docker)

```bash
docker compose up --build
```

Open [http://localhost:8000](http://localhost:8000). Done.

> The ONNX model files (`neural_art_transfer.onnx` + `.onnx.data`) must be in the project root. If you don't have them, see **Training** below.

## Manual Setup

### Run the web app (inference only)

```bash
pip install -r requirements-docker.txt
python inference.py
```

### Prepare Datasets (for training)

Training requires two datasets structured as follows:
```
src/datasets/
├── coco/
│   └── train2017/         # Content images
└── wikiart/               # Style images
```

1. **COCO 2017 (Content)**: Download the **2017 Train images** (18 GB) from the [COCO Dataset Download page](https://cocodataset.org/#download). Extract the images into `src/datasets/coco/train2017/`.
2. **WikiArt (Style)**: Run the helper script to download the dataset via KaggleHub:
   ```bash
   pip install kagglehub
   python src/download_data.py
   ```
   Copy the downloaded dataset contents from the cache path printed by the script into `src/datasets/wikiart/`.

### Train from scratch

```bash
pip install -r requirements.txt
cd src
python train.py
python save_as_onnx.py --checkpoint checkpoints/decoder_epoch_20.pth
```

Move the generated `.onnx` and `.onnx.data` files to the project root.

## Project Structure

```
├── inference.py                FastAPI server + ONNX inference
├── static/                     Web frontend
│   ├── index.html
│   ├── style.css
│   └── app.js
├── src/                        Training code
│   ├── model.py                ResNet50 encoder + decoder + full model
│   ├── train.py                Training loop
│   ├── utils.py                AdaIN, loss functions
│   ├── dataset.py              Content/style dataset loader
│   ├── save_as_onnx.py         Export trained model to ONNX
│   └── download_data.py        Download WikiArt dataset
├── Dockerfile
├── docker-compose.yml
├── requirements.txt            Everything (training + inference)
└── requirements-docker.txt     Inference only
```

## How It Works

A frozen ResNet50 (up to layer3) encodes both images into feature maps. AdaIN (Adaptive Instance Normalization) aligns the content features' channel-wise mean and variance to match the style image's statistics. A learned decoder reconstructs the stylized image from the blended features. The style strength slider controls how much of the style statistics are applied vs. keeping the original content features.

## GPU Support (Docker)

Add this to `docker-compose.yml` under the `app` service and switch `onnxruntime` to `onnxruntime-gpu` in `requirements-docker.txt`:

```yaml
deploy:
  resources:
    reservations:
      devices:
        - driver: nvidia
          count: 1
          capabilities: [gpu]
```

Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html).
