FROM jupyter/tensorflow-notebook:latest

USER root

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install NVIDIA drivers and CUDA support (if needed)
RUN apt-get update && apt-get install -y \
    nvidia-container-toolkit \
    && rm -rf /var/lib/apt/lists/*

USER $NB_UID

# Install Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt

# Install additional Jupyter extensions
RUN pip install jupyterlab-git ipywidgets

# Set working directory
WORKDIR /workspace

# Expose Jupyter port
EXPOSE 8888

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8888/api || exit 1

# Start Jupyter
CMD ["start-notebook.sh", "--NotebookApp.token='fine-tuning-llms'", "--NotebookApp.password=''", "--NotebookApp.allow_root=True"]
