# Use a slim Python base instead of full CUDA image
FROM python:3.10-slim

# Prevent Python from writing .pyc files and buffering stdout
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install CPU-only PyTorch + torchvision
# (use official pip wheels without CUDA)
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Install other dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files
WORKDIR /app
COPY . /app

# Expose Streamlit port
EXPOSE 8501

# Run the app
CMD ["streamlit", "run", "dcgan_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
