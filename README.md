# Neural Art Studio

High-quality neural style transfer using the optimization-based method originally proposed by Gatys et al. (2015). Drop in a content image and a style image, adjust the style strength slider, and watch the image optimize in the background. 

![preview](static/preview.png)

## Quick Start

### 1. Install dependencies

First, install the standard dependencies:

```bash
pip install -r requirements.txt
```

**For NVIDIA GPU Users (Highly Recommended):**
Neural style transfer runs much faster on a GPU. To enable GPU acceleration, install the CUDA-enabled version of PyTorch by running one of the commands below depending on your CUDA version:

```bash

# uninstall cpu version 
pip uninstall torch torchvision

```
```bash
# For CUDA 12.6
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```
```bash
# For CUDA 13.0
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

### 2. Run the app

```bash
python inference.py
```

Open [http://localhost:8000](http://localhost:8000). Pick a style, upload a photo, and click Generate.

> **Note**: Because this uses the Gatys optimization method, image generation takes time (around 10-30 seconds depending on your hardware) as it runs 100 steps of gradient descent using L-BFGS.

## Docker

```bash
docker compose up --build
```

## Project Structure

```
├── inference.py                FastAPI server + PyTorch L-BFGS optimization loop
├── static/                     Web frontend
│   ├── index.html
│   ├── style.css
│   ├── app.js
│   └── styles/                 5 built-in style presets
├── src/
│   ├── model.py                VGG19 feature extractor matching the Gatys paper
│   └── utils.py                Loss functions (Gram matrix, TV loss, etc.)
├── Dockerfile
├── docker-compose.yml
├── requirements.txt            Dependencies
└── requirements-docker.txt     Docker dependencies
```

## How It Works

Unlike fast style transfer networks (like AdaIN) that use a single forward pass, this method starts with the content image and **iteratively optimizes its pixels** over 100 steps using the L-BFGS algorithm. 

A pre-trained VGG19 network extracts feature maps at various depths. The optimizer works to minimize a combined loss function:
- **Content Loss**: Mean squared error between the `conv4_2` features of the generated image and the content image.
- **Style Loss**: Mean squared error between the Gram matrices of the `relu1_1`, `relu2_1`, `relu3_1`, `relu4_1`, and `relu5_1` features of the generated image and the style image.
- **Total Variation (TV) Loss**: Smooths the image to reduce high-frequency noise.

The style strength slider controls the ratio of style weight to content weight during optimization.

## GPU Support (Docker)

Add this to `docker-compose.yml` under the `app` service:

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
