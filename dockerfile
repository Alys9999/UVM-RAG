# GPU backend Dockerfile with:
# - CUDA 11.8 runtime
# - Python 3.13
# - torch==2.7.1 / torchvision==0.22.1 / torchaudio==2.7.1 (cu118)
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/models/hf \
    PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu118


# Build deps for Python, plus runtime libs you need
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential wget curl ca-certificates \
    libssl-dev zlib1g-dev libncurses5-dev libncursesw5-dev \
    libreadline-dev libffi-dev libsqlite3-dev libbz2-dev liblzma-dev \
    uuid-dev tk-dev \
    git ffmpeg libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

# Build and install Python 3.13.5
WORKDIR /tmp
RUN wget https://www.python.org/ftp/python/3.13.5/Python-3.13.5.tgz && \
    tar -xzf Python-3.13.5.tgz && \
    cd Python-3.13.5 && \
    ./configure --enable-optimizations --with-lto --prefix=/usr/local && \
    make -j"$(nproc)" && \
    make altinstall && \
    cd /tmp && rm -rf Python-3.13.5 Python-3.13.5.tgz

RUN python3.13 -m ensurepip && \
    python3.13 -m pip install --no-cache-dir --upgrade pip
    
RUN python3.13 -m pip install --no-cache-dir --upgrade pip



# ------------------------------------------------------------
# 2. Workdir and requirements
# ------------------------------------------------------------
WORKDIR /app

# Copy your pip-freeze requirements.txt into the image
COPY . .
ENV PG_CONN_STR="postgresql://postgres:postgres@uvm-postgres-cloud:5432/postgres"
# ------------------------------------------------------------
# 3. Install PyTorch stack with cu118 EXACTLY as requested
# ------------------------------------------------------------
RUN python3.13 -m pip install \
    torch==2.7.1 \
    torchvision==0.22.1 \
    torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu118

ENV UVM_RAG_AUTOWARM=1



ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501

EXPOSE 8501

ENTRYPOINT ["bash", "-lc"]

CMD ["python3.13 -m pip install -r requirements.txt && \
      python3.13 -c 'import backend' && \
      streamlit run app/ui_streamlit.py --server.address=${STREAMLIT_SERVER_ADDRESS} --server.port=${STREAMLIT_SERVER_PORT}"]
