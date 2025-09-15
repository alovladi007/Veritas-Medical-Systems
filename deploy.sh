#!/bin/bash

# KGAREVION Medical QA System Deployment Script
# This script sets up and deploys the complete fullstack application

set -e

echo "🚀 Starting KGAREVION Medical QA System Deployment..."

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Docker is installed
check_docker() {
    print_status "Checking Docker installation..."
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    if ! command -v docker-compose &> /dev/null; then
        print_error "Docker Compose is not installed. Please install Docker Compose first."
        exit 1
    fi
    
    print_success "Docker and Docker Compose are installed"
}

# Check if required ports are available
check_ports() {
    print_status "Checking if required ports are available..."
    
    ports=(3000 8000 5432 7474 7687 6379)
    for port in "${ports[@]}"; do
        if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1; then
            print_warning "Port $port is already in use. You may need to stop the service using this port."
        fi
    done
}

# Create environment file if it doesn't exist
setup_environment() {
    print_status "Setting up environment configuration..."
    
    if [ ! -f .env ]; then
        if [ -f env.example ]; then
            cp env.example .env
            print_success "Created .env file from env.example"
            print_warning "Please review and update the .env file with your specific configuration"
        else
            print_error "env.example file not found. Please create a .env file manually."
            exit 1
        fi
    else
        print_success ".env file already exists"
    fi
}

# Build and start services
start_services() {
    print_status "Building and starting services..."
    
    # Build images
    print_status "Building Docker images..."
    docker-compose build --no-cache
    
    # Start services
    print_status "Starting services..."
    docker-compose up -d
    
    print_success "Services started successfully"
}

# Wait for services to be ready
wait_for_services() {
    print_status "Waiting for services to be ready..."
    
    # Wait for PostgreSQL
    print_status "Waiting for PostgreSQL..."
    until docker-compose exec -T postgres pg_isready -U medical_qa > /dev/null 2>&1; do
        sleep 2
    done
    print_success "PostgreSQL is ready"
    
    # Wait for Neo4j
    print_status "Waiting for Neo4j..."
    until curl -s http://localhost:7474 > /dev/null 2>&1; do
        sleep 2
    done
    print_success "Neo4j is ready"
    
    # Wait for Redis
    print_status "Waiting for Redis..."
    until docker-compose exec -T redis redis-cli ping > /dev/null 2>&1; do
        sleep 2
    done
    print_success "Redis is ready"
    
    # Wait for backend API
    print_status "Waiting for backend API..."
    until curl -s http://localhost:8000/health > /dev/null 2>&1; do
        sleep 2
    done
    print_success "Backend API is ready"
    
    # Wait for frontend
    print_status "Waiting for frontend..."
    until curl -s http://localhost:3000 > /dev/null 2>&1; do
        sleep 2
    done
    print_success "Frontend is ready"
}

# Initialize database
init_database() {
    print_status "Initializing database..."
    
    # Wait a bit more for database to be fully ready
    sleep 5
    
    # Run database initialization
    docker-compose exec -T postgres psql -U medical_qa -d kgarevion -f /docker-entrypoint-initdb.d/init.sql
    
    print_success "Database initialized successfully"
}

# Show deployment status
show_status() {
    print_success "🎉 Deployment completed successfully!"
    echo ""
    echo "📋 Service URLs:"
    echo "   Frontend:     http://localhost:3000"
    echo "   Backend API:  http://localhost:8000"
    echo "   API Docs:     http://localhost:8000/docs"
    echo "   Neo4j:        http://localhost:7474"
    echo ""
    echo "🔐 Default Admin Credentials:"
    echo "   Username: admin"
    echo "   Password: admin123"
    echo ""
    echo "📊 To view logs:"
    echo "   docker-compose logs -f"
    echo ""
    echo "🛑 To stop services:"
    echo "   docker-compose down"
    echo ""
    echo "🔄 To restart services:"
    echo "   docker-compose restart"
}

# Main deployment function
main() {
    print_status "Starting KGAREVION Medical QA System deployment..."
    
    check_docker
    check_ports
    setup_environment
    start_services
    wait_for_services
    init_database
    show_status
}

# Handle script arguments
case "${1:-}" in
    "stop")
        print_status "Stopping KGAREVION services..."
        docker-compose down
        print_success "Services stopped"
        ;;
    "restart")
        print_status "Restarting KGAREVION services..."
        docker-compose restart
        print_success "Services restarted"
        ;;
    "logs")
        docker-compose logs -f
        ;;
    "status")
        docker-compose ps
        ;;
    "clean")
        print_status "Cleaning up KGAREVION deployment..."
        docker-compose down -v
        docker system prune -f
        print_success "Cleanup completed"
        ;;
    *)
        main
        ;;
esac
