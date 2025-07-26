# Part 5: Deployment with Ollama and Docker

This directory contains complete deployment solutions for production-ready LLM services using Ollama and Docker.

## 🎯 Features

- **Ollama Integration**: Local model serving with GGUF format support
- **FastAPI Backend**: High-performance REST API with authentication
- **Streamlit Frontend**: Interactive web interface for model interaction
- **Docker Orchestration**: Complete containerized deployment stack
- **Load Balancing**: Nginx configuration for traffic distribution
- **Monitoring Ready**: Prometheus metrics and health checks

## 🚀 Quick Start

```bash
# Deploy the complete stack
./scripts/deploy.sh

# Or use Docker Compose directly
docker-compose up -d

# Check service status
docker-compose ps

# View logs
docker-compose logs -f api
```

## 📁 Directory Contents

- `src/` - API and web application source code
- `docker/` - Docker configurations and Dockerfiles
- `configs/` - Service configuration files
- `scripts/` - Deployment and management scripts
- `docs/` - Deployment guides and API documentation

## 🔧 Key Components

### FastAPI Service (`src/api/`)
- RESTful API endpoints for model inference
- JWT authentication and authorization
- Request validation and rate limiting
- Prometheus metrics integration
- Health checks and monitoring

### Streamlit Interface (`src/web/`)
- Interactive chat interface
- Model comparison tools
- Usage analytics dashboard
- Admin configuration panel

### Ollama Integration (`src/ollama/`)
- Model conversion to GGUF format
- Ollama service management
- Performance optimization
- Model switching capabilities

### Docker Stack (`docker/`)
- Multi-service orchestration
- Development and production configurations
- Health checks and restart policies
- Volume management for persistence

## 🌐 API Endpoints

### Core Endpoints
- `POST /api/v1/generate` - Generate text completions
- `POST /api/v1/chat` - Interactive chat completions
- `GET /api/v1/models` - List available models
- `POST /api/v1/models/load` - Load specific model
- `GET /api/v1/health` - Service health check

### Management Endpoints
- `GET /api/v1/metrics` - Prometheus metrics
- `GET /api/v1/status` - Service status and statistics
- `POST /api/v1/admin/reload` - Reload configuration
- `GET /api/v1/docs` - Interactive API documentation

## 🐳 Docker Services

The deployment includes:

- **API Service**: FastAPI application server
- **Web Service**: Streamlit web interface
- **Ollama Service**: Model serving and inference
- **Nginx Service**: Load balancer and reverse proxy
- **Redis Service**: Caching and session storage
- **Prometheus**: Metrics collection
- **Grafana**: Monitoring dashboards

## 🔒 Security Features

- **JWT Authentication**: Secure API access
- **Rate Limiting**: Request throttling and abuse prevention
- **Input Validation**: Request sanitization and validation
- **CORS Configuration**: Cross-origin request handling
- **SSL/TLS Support**: HTTPS encryption ready

## 📖 Related Blog Post

[Part 5: Deployment with Ollama and Docker](https://saptak.github.io/2025/07/25/fine-tuning-small-llms-part5-deployment/)

## 🔗 Usage Examples

See the `docs/` directory for:
- Complete deployment guides
- API usage examples
- Configuration options
- Troubleshooting guides
- Performance tuning tips
