# GPU runtime matching torch 2.7.1+cu118
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu118

# Install Python 3.13 and basics
RUN apt-get update && \
    apt-get install -y --no-install-recommends software-properties-common curl ca-certificates && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.13 python3.13-venv python3.13-distutils python3-pip \
        git ffmpeg libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Upgrade pip for 3.13
RUN python3.13 -m pip install --no-cache-dir --upgrade pip

WORKDIR /app

# Copy your frozen requirements
COPY requirements.txt .

# Install exactly the frozen set (torch/torchvision/torchaudio use cu118 index via PIP_EXTRA_INDEX_URL)
RUN python3.13 -m pip install --no-cache-dir -r requirements.txt

# Copy app code (adjust paths as needed)
COPY app app
COPY .env .env

# Expose Streamlit port (change if you use something else)
EXPOSE 8501

# Default CMD; change to your launcher if different
CMD ["streamlit", "run", "app/ui_streamlit.py", "--server.address=0.0.0.0", "--server.port=8501"]
