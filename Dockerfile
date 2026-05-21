FROM python:3.11-slim

WORKDIR /app

COPY requirements-docker.txt .
RUN pip install --no-cache-dir -r requirements-docker.txt

COPY inference.py .
COPY neural_art_transfer.onnx .
COPY neural_art_transfer.onnx.data .
COPY static/ static/

EXPOSE 8000

CMD ["uvicorn", "inference:app", "--host", "0.0.0.0", "--port", "8000"]
