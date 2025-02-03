# Use specified NVIDIA CUDA base image
FROM nvidia/cuda:11.3.1-cudnn8-runtime-ubuntu20.04

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy configuration files first
COPY environment.yml /tmp/environment.yml

# Copy scripts directory
COPY src/scripts /app/src/scripts

# Make setup script executable and run it
RUN chmod +x /app/src/scripts/setup.sh && \
    /app/src/scripts/setup.sh

# Set default environment variables
ENV PATH /opt/conda/envs/diffprivate/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV CUDA_VISIBLE_DEVICES=0

WORKDIR /app
CMD ["python"]
