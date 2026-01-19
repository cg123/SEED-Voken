FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-devel

# Set working directory
WORKDIR /workspace/SEED-Voken

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONFAULTHANDLER=1

# Default command (will be overridden by K8s)
CMD ["python", "main.py", "--help"]
