#!/bin/bash
# Quick Start Script for Fine-Tuning Small LLMs
# Usage: ./scripts/quick_start.sh

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${GREEN}🚀 Fine-Tuning Small LLMs - Quick Start${NC}"
echo "=================================================="

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo -e "${RED}❌ Docker is not running. Please start Docker Desktop.${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Docker is running${NC}"

# Check if .env file exists
if [ ! -f ".env" ]; then
    echo -e "${YELLOW}📝 Creating .env file from template...${NC}"
    cp .env.example .env
    echo -e "${YELLOW}⚠️  Please review and update .env file with your settings${NC}"
fi

# Create necessary directories
echo -e "${GREEN}📁 Creating directories...${NC}"
mkdir -p {data/datasets,models,logs,backups}

# Start the services
echo -e "${GREEN}🐳 Starting services...${NC}"
docker-compose up -d

# Wait for services to be ready
echo -e "${GREEN}⏳ Waiting for services to start...${NC}"
sleep 30

# Check service health
echo -e "${GREEN}🔍 Checking service health...${NC}"

services=("ollama:11434" "redis:6379")
for service in "${services[@]}"; do
    service_name=$(echo $service | cut -d':' -f1)
    port=$(echo $service | cut -d':' -f2)
    
    if curl -f -s "http://localhost:$port" >/dev/null 2>&1; then
        echo -e "${GREEN}✅ $service_name is ready${NC}"
    else
        echo -e "${YELLOW}⚠️  $service_name may still be starting...${NC}"
    fi
done

echo ""
echo -e "${GREEN}🎉 Quick start completed!${NC}"
echo ""
echo "📋 Available Services:"
echo "  🤖 API Documentation: http://localhost:8000/docs"
echo "  💬 Web Interface:     http://localhost:8501"
echo "  📊 Grafana:           http://localhost:3000 (admin/admin123)"
echo "  📈 Prometheus:        http://localhost:9090"
echo ""
echo "🔗 Next Steps:"
echo "  1. Create training data: python part2-data-preparation/src/dataset_creation.py"
echo "  2. Train a model:       python part3-training/src/fine_tune_model.py --config part3-training/configs/sql_expert.yaml"
echo "  3. Evaluate model:      python part4-evaluation/src/run_evaluation.py --model-path ./models/sql-expert-merged"
echo "  4. Deploy to production: Follow Part 6 production guide"
echo ""
echo -e "${GREEN}Happy fine-tuning! 🚀${NC}"
