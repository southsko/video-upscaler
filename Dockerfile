FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install Python 3.12, ffmpeg, and essentials (Ubuntu 24.04 ships 3.12)
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-venv python3-pip ffmpeg \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Create a venv so pip doesn't fight with the system Python
RUN python3 -m venv /app/.venv
ENV PATH="/app/.venv/bin:$PATH"

RUN pip install --upgrade pip

# Install PyTorch with CUDA first
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

# Install the rest of the dependencies
COPY requirements.txt .
RUN pip install -r requirements.txt

# Copy source code (includes our fixes: spatial-aq, live preview, Linux browse)
COPY upscale_video.py server.py ./
COPY webui/ ./webui/
COPY pyproject.toml .

# Models directory (persisted via volume)
RUN mkdir -p /app/models

EXPOSE 8848

# Token can be set via UPSCALE_TOKEN env var, otherwise auto-generated
CMD ["python3", "upscale_video.py", "--serve", "--host", "0.0.0.0"]
