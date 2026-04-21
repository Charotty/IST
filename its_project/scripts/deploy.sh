#!/bin/bash
# Production Deployment Script for Intelligent Trading System

set -e

echo "=== Intelligent Trading System - Production Deployment ==="

# Configuration
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COMPOSE_FILE="${PROJECT_ROOT}/docker-compose.yml"
CONFIG_FILE="${PROJECT_ROOT}/config.prod.yaml"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Functions
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    echo "Checking prerequisites..."
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    print_success "Docker is installed"
    
    # Check Docker Compose
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    print_success "Docker Compose is installed"
    
    # Check if config file exists
    if [ ! -f "$CONFIG_FILE" ]; then
        print_error "Production config file not found: $CONFIG_FILE"
        exit 1
    fi
    print_success "Production config file found"
    
    # Check environment variables
    if [ -z "$BINANCE_API_KEY" ]; then
        print_warning "BINANCE_API_KEY environment variable not set"
    fi
    if [ -z "$BINANCE_API_SECRET" ]; then
        print_warning "BINANCE_API_SECRET environment variable not set"
    fi
}

# Setup directories
setup_directories() {
    echo "Setting up directories..."
    
    mkdir -p "${PROJECT_ROOT}/data"
    mkdir -p "${PROJECT_ROOT}/logs"
    mkdir -p "${PROJECT_ROOT}/cache"
    mkdir -p "${PROJECT_ROOT}/monitoring/prometheus"
    mkdir -p "${PROJECT_ROOT}/monitoring/grafana/dashboards"
    mkdir -p "${PROJECT_ROOT}/monitoring/grafana/datasources"
    
    print_success "Directories created"
}

# Build Docker images
build_images() {
    echo "Building Docker images..."
    
    cd "$PROJECT_ROOT"
    docker-compose build
    
    print_success "Docker images built"
}

# Start services
start_services() {
    echo "Starting services..."
    
    cd "$PROJECT_ROOT"
    docker-compose up -d
    
    print_success "Services started"
}

# Stop services
stop_services() {
    echo "Stopping services..."
    
    cd "$PROJECT_ROOT"
    docker-compose down
    
    print_success "Services stopped"
}

# Restart services
restart_services() {
    echo "Restarting services..."
    
    cd "$PROJECT_ROOT"
    docker-compose restart
    
    print_success "Services restarted"
}

# Check service health
check_health() {
    echo "Checking service health..."
    
    cd "$PROJECT_ROOT"
    docker-compose ps
    
    echo ""
    echo "Service URLs:"
    echo "  - Web Dashboard: http://localhost:5000"
    echo "  - Grafana: http://localhost:3000 (admin/admin)"
    echo "  - Prometheus: http://localhost:9090"
}

# View logs
view_logs() {
    local service=$1
    
    cd "$PROJECT_ROOT"
    if [ -z "$service" ]; then
        docker-compose logs -f
    else
        docker-compose logs -f "$service"
    fi
}

# Main deployment function
deploy() {
    echo "Starting deployment..."
    
    check_prerequisites
    setup_directories
    build_images
    start_services
    
    echo ""
    print_success "Deployment completed successfully!"
    echo ""
    echo "Next steps:"
    echo "  1. Check service health: ./scripts/deploy.sh health"
    echo "  2. View logs: ./scripts/deploy.sh logs"
    echo "  3. Access dashboard: http://localhost:5000"
}

# Main script
case "${1:-deploy}" in
    deploy)
        deploy
        ;;
    stop)
        stop_services
        ;;
    start)
        start_services
        ;;
    restart)
        restart_services
        ;;
    health)
        check_health
        ;;
    logs)
        view_logs "${2:-}"
        ;;
    rebuild)
        stop_services
        build_images
        start_services
        ;;
    *)
        echo "Usage: $0 {deploy|stop|start|restart|health|logs|rebuild}"
        echo ""
        echo "Commands:"
        echo "  deploy    - Deploy the system (default)"
        echo "  stop      - Stop all services"
        echo "  start     - Start all services"
        echo "  restart   - Restart all services"
        echo "  health    - Check service health"
        echo "  logs      - View logs (optional: specify service name)"
        echo "  rebuild   - Rebuild and restart services"
        exit 1
        ;;
esac
