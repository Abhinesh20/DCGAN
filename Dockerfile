# Use official PyTorch image with CUDA (if GPU needed)
FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy project files
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install streamlit torchvision

# Expose port for Streamlit
EXPOSE 8501

# Command to run app
CMD ["streamlit", "run", "dcgan_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
