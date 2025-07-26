#!/bin/bash
# Production Setup Script
# From: Fine-Tuning Small LLMs with Docker Desktop - Part 6

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

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

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$(dirname "$SCRIPT_DIR")")"

print_header "🚀 Production Monitoring & Optimization Setup"
echo "============================================================"

# Check prerequisites
check_prerequisites() {
    print_status "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        exit 1
    fi
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed"
        exit 1
    fi
    
    # Check available disk space
    available_space=$(df / | tail -1 | awk '{print $4}')
    if [ "$available_space" -lt 10485760 ]; then  # 10GB in KB
        print_warning "Less than 10GB disk space available"
    fi
    
    print_status "Prerequisites check passed"
}

# Setup monitoring stack
setup_monitoring() {
    print_status "Setting up monitoring stack..."
    
    cd "$PROJECT_ROOT"
    
    # Create monitoring directories
    mkdir -p {prometheus_data,grafana_data,alertmanager_data}
    mkdir -p logs/{prometheus,grafana,alertmanager}
    
    # Set permissions
    sudo chown -R 472:472 grafana_data 2>/dev/null || true
    sudo chown -R 65534:65534 prometheus_data 2>/dev/null || true
    
    # Create Grafana datasource configuration
    mkdir -p part6-production/monitoring/grafana/datasources
    cat > part6-production/monitoring/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

    # Create Grafana dashboard configuration
    mkdir -p part6-production/monitoring/grafana/dashboards
    cat > part6-production/monitoring/grafana/dashboards/dashboard.yml << EOF
apiVersion: 1

providers:
  - name: 'default'
    orgId: 1
    folder: ''
    type: file
    disableDeletion: false
    updateIntervalSeconds: 10
    allowUiUpdates: true
    options:
      path: /etc/grafana/provisioning/dashboards
EOF

    # Start monitoring stack
    docker-compose -f docker-compose.yml up -d prometheus grafana
    
    print_status "Monitoring stack started"
    print_status "Grafana: http://localhost:3000 (admin/admin123)"
    print_status "Prometheus: http://localhost:9090"
}

# Setup alerting
setup_alerting() {
    print_status "Setting up alerting rules..."
    
    # Create alert rules
    cat > part6-production/monitoring/alert_rules.yml << EOF
groups:
  - name: llm_alerts
    rules:
      # High response time alert
      - alert: HighResponseTime
        expr: histogram_quantile(0.95, rate(http_request_duration_seconds_bucket[5m])) > 2
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High response time detected"
          description: "95th percentile response time is {{ \$value }}s"

      # High error rate alert
      - alert: HighErrorRate
        expr: rate(http_requests_total{status=~"5.."}[5m]) / rate(http_requests_total[5m]) > 0.1
        for: 3m
        labels:
          severity: critical
        annotations:
          summary: "High error rate detected"
          description: "Error rate is {{ \$value | humanizePercentage }}"

      # Model inference queue backup
      - alert: ModelQueueBackup
        expr: model_inference_queue_length > 100
        for: 2m
        labels:
          severity: warning
        annotations:
          summary: "Model inference queue backing up"
          description: "Queue length is {{ \$value }} requests"

      # High memory usage
      - alert: HighMemoryUsage
        expr: (node_memory_MemTotal_bytes - node_memory_MemAvailable_bytes) / node_memory_MemTotal_bytes > 0.9
        for: 5m
        labels:
          severity: warning
        annotations:
          summary: "High memory usage"
          description: "Memory usage is {{ \$value | humanizePercentage }}"

      # Disk space low
      - alert: DiskSpaceLow
        expr: (node_filesystem_avail_bytes / node_filesystem_size_bytes) < 0.1
        for: 10m
        labels:
          severity: critical
        annotations:
          summary: "Low disk space"
          description: "Disk space is {{ \$value | humanizePercentage }} available"

      # Container down
      - alert: ContainerDown
        expr: up == 0
        for: 1m
        labels:
          severity: critical
        annotations:
          summary: "Container is down"
          description: "Container {{ \$labels.instance }} is down"
EOF

    print_status "Alert rules configured"
}

# Setup log aggregation
setup_logging() {
    print_status "Setting up log aggregation..."
    
    # Create log aggregation configuration
    mkdir -p part6-production/monitoring/filebeat
    cat > part6-production/monitoring/filebeat/filebeat.yml << EOF
filebeat.inputs:
- type: log
  enabled: true
  paths:
    - /var/log/containers/*.log
  fields:
    service: llm-api
  fields_under_root: true

- type: log
  enabled: true
  paths:
    - /app/logs/*.log
  fields:
    service: application
  fields_under_root: true

output.elasticsearch:
  hosts: ["elasticsearch:9200"]
  index: "llm-logs-%{+yyyy.MM.dd}"

setup.template.name: "llm-logs"
setup.template.pattern: "llm-logs-*"
setup.template.enabled: true

logging.level: info
logging.to_files: true
logging.files:
  path: /var/log/filebeat
  name: filebeat
  keepfiles: 7
  permissions: 0644
EOF

    print_status "Log aggregation configured"
}

# Setup performance optimization
setup_optimization() {
    print_status "Setting up performance optimization..."
    
    # Create performance monitoring script
    cat > part6-production/scripts/performance_monitor.py << 'EOF'
#!/usr/bin/env python3
"""
Performance monitoring and optimization script
"""

import psutil
import docker
import time
import json
import logging
from typing import Dict, Any

class PerformanceMonitor:
    def __init__(self):
        self.docker_client = docker.from_env()
        self.thresholds = {
            'cpu_percent': 80,
            'memory_percent': 85,
            'disk_percent': 90
        }
    
    def get_system_metrics(self) -> Dict[str, Any]:
        """Get current system metrics"""
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory usage
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        
        # Disk usage
        disk = psutil.disk_usage('/')
        disk_percent = (disk.used / disk.total) * 100
        
        return {
            'timestamp': time.time(),
            'cpu_percent': cpu_percent,
            'memory_percent': memory_percent,
            'disk_percent': disk_percent,
            'memory_available_gb': memory.available / (1024**3),
            'disk_free_gb': disk.free / (1024**3)
        }
    
    def optimize_containers(self):
        """Optimize container resource allocation"""
        
        containers = self.docker_client.containers.list()
        
        for container in containers:
            try:
                stats = container.stats(stream=False)
                
                # Calculate CPU usage
                cpu_delta = stats['cpu_stats']['cpu_usage']['total_usage'] - \
                           stats['precpu_stats']['cpu_usage']['total_usage']
                system_delta = stats['cpu_stats']['system_cpu_usage'] - \
                              stats['precpu_stats']['system_cpu_usage']
                
                cpu_percent = (cpu_delta / system_delta) * 100 if system_delta > 0 else 0
                
                # Memory usage
                memory_usage = stats['memory_stats']['usage']
                memory_limit = stats['memory_stats']['limit']
                memory_percent = (memory_usage / memory_limit) * 100
                
                print(f"Container {container.name}: CPU {cpu_percent:.1f}%, Memory {memory_percent:.1f}%")
                
                # Optimization recommendations
                if cpu_percent > 90:
                    print(f"  🔧 Recommendation: Scale up {container.name} (high CPU)")
                
                if memory_percent > 90:
                    print(f"  🔧 Recommendation: Increase memory for {container.name}")
                
            except Exception as e:
                logging.error(f"Error analyzing container {container.name}: {e}")
    
    def run_monitoring_loop(self, interval: int = 60):
        """Run continuous monitoring"""
        
        print("🔄 Starting performance monitoring...")
        
        while True:
            try:
                # Get system metrics
                metrics = self.get_system_metrics()
                
                # Log metrics
                print(f"System: CPU {metrics['cpu_percent']:.1f}%, "
                      f"Memory {metrics['memory_percent']:.1f}%, "
                      f"Disk {metrics['disk_percent']:.1f}%")
                
                # Check thresholds and alert
                for metric, threshold in self.thresholds.items():
                    if metrics[metric] > threshold:
                        print(f"⚠️  WARNING: {metric} is {metrics[metric]:.1f}% (threshold: {threshold}%)")
                
                # Optimize containers
                self.optimize_containers()
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                print("🛑 Monitoring stopped")
                break
            except Exception as e:
                logging.error(f"Monitoring error: {e}")
                time.sleep(interval)

if __name__ == "__main__":
    monitor = PerformanceMonitor()
    monitor.run_monitoring_loop()
EOF

    chmod +x part6-production/scripts/performance_monitor.py
    
    print_status "Performance optimization configured"
}

# Setup backup system
setup_backup() {
    print_status "Setting up backup system..."
    
    # Create backup script
    cat > part6-production/scripts/backup_system.sh << 'EOF'
#!/bin/bash
# Automated backup system

BACKUP_DIR="/backups"
S3_BUCKET="llm-backups"
RETENTION_DAYS=30

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup models
backup_models() {
    echo "📦 Backing up models..."
    tar -czf "$BACKUP_DIR/models_$(date +%Y%m%d_%H%M%S).tar.gz" ./models/
}

# Backup data
backup_data() {
    echo "📦 Backing up data..."
    tar -czf "$BACKUP_DIR/data_$(date +%Y%m%d_%H%M%S).tar.gz" ./data/
}

# Backup configurations
backup_configs() {
    echo "📦 Backing up configurations..."
    tar -czf "$BACKUP_DIR/configs_$(date +%Y%m%d_%H%M%S).tar.gz" \
        docker-compose.yml .env part*/configs/
}

# Upload to S3 (if configured)
upload_to_s3() {
    if command -v aws &> /dev/null && [ ! -z "$AWS_ACCESS_KEY_ID" ]; then
        echo "☁️  Uploading to S3..."
        aws s3 sync $BACKUP_DIR s3://$S3_BUCKET/backups/
    fi
}

# Cleanup old backups
cleanup_old_backups() {
    echo "🧹 Cleaning up old backups..."
    find $BACKUP_DIR -name "*.tar.gz" -mtime +$RETENTION_DAYS -delete
}

# Main backup function
main() {
    echo "🚀 Starting backup process..."
    
    backup_models
    backup_data
    backup_configs
    upload_to_s3
    cleanup_old_backups
    
    echo "✅ Backup completed successfully"
}

main "$@"
EOF

    chmod +x part6-production/scripts/backup_system.sh
    
    print_status "Backup system configured"
}

# Main setup function
main() {
    check_prerequisites
    setup_monitoring
    setup_alerting
    setup_logging
    setup_optimization
    setup_backup
    
    print_header "✅ Production setup completed!"
    echo ""
    echo "🌐 Service URLs:"
    echo "  📊 Grafana:     http://localhost:3000 (admin/admin123)"
    echo "  📈 Prometheus:  http://localhost:9090"
    echo "  🔍 API Docs:    http://localhost:8000/docs"
    echo ""
    echo "🔧 Management Commands:"
    echo "  Monitor performance: python part6-production/scripts/performance_monitor.py"
    echo "  Run backup:         ./part6-production/scripts/backup_system.sh"
    echo "  View logs:          docker-compose logs -f"
    echo ""
    echo "📊 Next Steps:"
    echo "  1. Configure alert destinations in Grafana"
    echo "  2. Set up SSL certificates for production"
    echo "  3. Configure cloud storage for backups"
    echo "  4. Review and adjust monitoring thresholds"
    echo ""
    print_status "Production environment is ready! 🚀"
}

main "$@"
