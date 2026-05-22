FROM python:3.11-slim

WORKDIR /app

ARG DEVICE=cpu

COPY requirements-docker.txt .
RUN if [ "$DEVICE" = "gpu" ]; then \
        pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cu13; \
    else \
        pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu; \
    fi && \
    pip install --no-cache-dir -r requirements-docker.txt

COPY app.py inference.py ./ 
COPY src/model.py src/model.py
COPY src/utils.py src/utils.py
COPY static/ static/

EXPOSE 8000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
