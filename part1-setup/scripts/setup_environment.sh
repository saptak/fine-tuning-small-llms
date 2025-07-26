#!/bin/bash
# Complete Environment Setup Script
# From: Fine-Tuning Small LLMs with Docker Desktop - Part 1

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_header() {
    echo -e "${BLUE}[SETUP]${NC} $1"
}

# Get the directory of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

print_header "🚀 Fine-Tuning Small LLMs - Environment Setup"
echo "============================================================"
echo "This script will set up your complete development environment"
echo "Project root: $PROJECT_ROOT"
echo ""

# Check if running on supported OS
check_os() {
    print_status "Checking operating system..."
    
    case "$(uname -s)" in
        Darwin*)
            OS="macOS"
            print_status "Detected: macOS"
            ;;
        Linux*)
            OS="Linux"
            print_status "Detected: Linux"
            ;;
        CYGWIN*|MINGW*|MSYS*)
            OS="Windows"
            print_status "Detected: Windows"
            ;;
        *)
            print_error "Unsupported operating system: $(uname -s)"
            exit 1
            ;;
    esac
}

# Install Docker Desktop
install_docker() {
    print_status "Checking Docker installation..."
    
    if command -v docker &> /dev/null; then
        DOCKER_VERSION=$(docker --version)
        print_status "Docker already installed: $DOCKER_VERSION"
        return 0
    fi
    
    print_status "Installing Docker Desktop..."
    
    case $OS in
        "macOS")
            if command -v brew &> /dev/null; then
                brew install --cask docker
            else
                print_warning "Homebrew not found. Please install Docker Desktop manually:"
                print_warning "https://www.docker.com/products/docker-desktop/"
                return 1
            fi
            ;;
        "Linux")
            # Install Docker Engine
            curl -fsSL https://get.docker.com -o get-docker.sh
            sh get-docker.sh
            sudo usermod -aG docker $USER
            rm get-docker.sh
            
            # Install Docker Compose
            sudo curl -L "https://github.com/docker/compose/releases/download/v2.20.0/docker-compose-$(uname -s)-$(uname -m)" -o /usr/local/bin/docker-compose
            sudo chmod +x /usr/local/bin/docker-compose
            ;;
        "Windows")
            print_warning "Please install Docker Desktop manually for Windows:"
            print_warning "https://www.docker.com/products/docker-desktop/"
            return 1
            ;;
    esac
    
    print_status "Docker installation completed"
}

# Setup Python environment
setup_python() {
    print_status "Setting up Python environment..."
    
    # Check Python version
    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 --version | cut -d' ' -f2)
        print_status "Python found: $PYTHON_VERSION"
    else
        print_error "Python 3 not found. Please install Python 3.10+"
        return 1
    fi
    
    # Create virtual environment
    if [ ! -d "$PROJECT_ROOT/venv" ]; then
        print_status "Creating virtual environment..."
        python3 -m venv "$PROJECT_ROOT/venv"
    fi
    
    # Activate virtual environment
    source "$PROJECT_ROOT/venv/bin/activate"
    
    # Upgrade pip
    print_status "Upgrading pip..."
    pip install --upgrade pip
    
    # Install requirements
    if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
        print_status "Installing Python dependencies..."
        pip install -r "$PROJECT_ROOT/requirements.txt"
    else
        print_status "Installing basic dependencies..."
        pip install torch torchvision torchaudio
        pip install transformers datasets
        pip install jupyter notebook
        pip install pandas numpy matplotlib seaborn
        pip install docker-py
        pip install psutil
    fi
    
    print_status "Python environment setup completed"
}

# Setup CUDA support (if NVIDIA GPU available)
setup_cuda() {
    print_status "Checking for NVIDIA GPU..."
    
    if command -v nvidia-smi &> /dev/null; then
        GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)
        print_status "NVIDIA GPU detected: $GPU_INFO"
        
        # Check CUDA installation
        if command -v nvcc &> /dev/null; then
            CUDA_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -c2-)
            print_status "CUDA already installed: $CUDA_VERSION"
        else
            print_warning "CUDA not found. For optimal performance, install CUDA:"
            print_warning "https://developer.nvidia.com/cuda-downloads"
        fi
    else
        print_warning "No NVIDIA GPU detected. Training will use CPU (slower)"
    fi
}

# Create project structure
create_project_structure() {
    print_status "Creating project structure..."
    
    cd "$PROJECT_ROOT"
    
    # Create directories if they don't exist
    mkdir -p {data,models,logs,backups}
    mkdir -p data/{datasets,processed,raw}
    mkdir -p models/{checkpoints,final,gguf}
    mkdir -p logs/{training,api,monitoring}
    
    print_status "Project structure created"
}

# Create configuration files
create_config_files() {
    print_status "Creating configuration files..."
    
    # Create .env file if it doesn't exist
    if [ ! -f "$PROJECT_ROOT/.env" ]; then
        cat > "$PROJECT_ROOT/.env" << EOF
# Environment Configuration
ENVIRONMENT=development

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
API_TOKEN=demo-token-12345

# Model Configuration
MODEL_NAME=sql-expert
MAX_TOKENS=256
TEMPERATURE=0.7

# Database Configuration
DATABASE_URL=sqlite:///./data/app.db

# Monitoring Configuration
PROMETHEUS_URL=http://localhost:9090
GRAFANA_URL=http://localhost:3000
GRAFANA_ADMIN_PASSWORD=admin123

# Ollama Configuration
OLLAMA_HOST=localhost
OLLAMA_PORT=11434

# Training Configuration
WANDB_PROJECT=llm-fine-tuning
WANDB_ENTITY=your-username

# Security
JWT_SECRET_KEY=your-secret-key-here
ENCRYPTION_KEY=your-encryption-key-here

# AWS Configuration (for backups)
AWS_ACCESS_KEY_ID=your-access-key
AWS_SECRET_ACCESS_KEY=your-secret-key
AWS_REGION=us-west-2
S3_BUCKET=your-backup-bucket
EOF
        print_status "Created .env file with default configuration"
        print_warning "Please review and update .env file with your settings"
    fi
    
    # Create requirements.txt if it doesn't exist
    if [ ! -f "$PROJECT_ROOT/requirements.txt" ]; then
        cat > "$PROJECT_ROOT/requirements.txt" << EOF
# Core ML Libraries
torch>=2.0.0
torchvision>=0.15.0
torchaudio>=2.0.0
transformers>=4.30.0
datasets>=2.12.0
accelerate>=0.20.0

# Unsloth for efficient training
unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git

# Training and Evaluation
wandb>=0.15.0
evaluate>=0.4.0
rouge-score>=0.1.2
nltk>=3.8
scikit-learn>=1.3.0

# API and Web Framework
fastapi>=0.100.0
uvicorn[standard]>=0.22.0
streamlit>=1.24.0
gradio>=3.35.0

# Data Processing
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
Pillow>=9.5.0

# Database and Storage
sqlalchemy>=2.0.0
alembic>=1.11.0
redis>=4.5.0
boto3>=1.26.0

# Monitoring and Logging
prometheus-client>=0.16.0
structlog>=23.1.0
python-multipart>=0.0.6

# Security
python-jose[cryptography]>=3.3.0
passlib[bcrypt]>=1.7.4
python-multipart>=0.0.6

# Utilities
python-dotenv>=1.0.0
click>=8.1.0
tqdm>=4.65.0
psutil>=5.9.0
docker>=6.1.0
pyyaml>=6.0

# Development
pytest>=7.4.0
black>=23.3.0
isort>=5.12.0
flake8>=6.0.0
mypy>=1.4.0
jupyter>=1.0.0
notebook>=6.5.0
EOF
        print_status "Created requirements.txt"
    fi
}

# Setup development environment
setup_dev_environment() {
    print_status "Setting up development environment..."
    
    # Start Jupyter notebook in background
    if command -v jupyter &> /dev/null; then
        print_status "Starting Jupyter notebook server..."
        nohup jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root > "$PROJECT_ROOT/logs/jupyter.log" 2>&1 &
        echo $! > "$PROJECT_ROOT/jupyter.pid"
        print_status "Jupyter notebook started on http://localhost:8888"
    fi
}

# Verify installation
verify_installation() {
    print_status "Verifying installation..."
    
    # Run system check
    if [ -f "$PROJECT_ROOT/part1-setup/src/system_check.py" ]; then
        print_status "Running system requirements check..."
        cd "$PROJECT_ROOT"
        python part1-setup/src/system_check.py
    else
        print_warning "System check script not found"
    fi
}

# Main setup function
main() {
    check_os
    install_docker
    setup_python
    setup_cuda
    create_project_structure
    create_config_files
    setup_dev_environment
    verify_installation
    
    print_header "🎉 Setup Complete!"
    echo ""
    echo "Next steps:"
    echo "1. Review and update the .env file with your configuration"
    echo "2. Start the development environment: docker-compose up -d"
    echo "3. Access Jupyter notebook: http://localhost:8888"
    echo "4. Follow Part 2 of the blog series for data preparation"
    echo ""
    echo "Useful commands:"
    echo "  - Activate Python environment: source venv/bin/activate"
    echo "  - Run system check: python part1-setup/src/system_check.py"
    echo "  - View logs: tail -f logs/*.log"
    echo ""
    print_status "Happy fine-tuning! 🚀"
}

# Run main function
main "$@"
