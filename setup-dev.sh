#!/bin/bash

# KGAREVION Medical QA System Development Setup Script
# This script sets up the development environment for local development

set -e

echo "🛠️  Setting up KGAREVION Medical QA System for Development..."

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

# Check if Python is installed
check_python() {
    print_status "Checking Python installation..."
    if ! command -v python3 &> /dev/null; then
        print_error "Python 3 is not installed. Please install Python 3.10+ first."
        exit 1
    fi
    
    python_version=$(python3 -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
    if [[ $(echo "$python_version 3.10" | awk '{print ($1 >= $2)}') == 0 ]]; then
        print_error "Python 3.10+ is required. Current version: $python_version"
        exit 1
    fi
    
    print_success "Python $python_version is installed"
}

# Check if Node.js is installed
check_node() {
    print_status "Checking Node.js installation..."
    if ! command -v node &> /dev/null; then
        print_error "Node.js is not installed. Please install Node.js 18+ first."
        exit 1
    fi
    
    node_version=$(node -v | cut -d'v' -f2)
    if [[ $(echo "$node_version 18.0" | awk '{print ($1 >= $2)}') == 0 ]]; then
        print_error "Node.js 18+ is required. Current version: $node_version"
        exit 1
    fi
    
    print_success "Node.js $node_version is installed"
}

# Setup Python virtual environment
setup_python_env() {
    print_status "Setting up Python virtual environment..."
    
    if [ ! -d "venv" ]; then
        python3 -m venv venv
        print_success "Created Python virtual environment"
    else
        print_success "Python virtual environment already exists"
    fi
    
    source venv/bin/activate
    print_status "Activated virtual environment"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install Python dependencies
    print_status "Installing Python dependencies..."
    pip install -r requirements.txt
    
    print_success "Python dependencies installed"
}

# Setup Node.js dependencies
setup_node_env() {
    print_status "Setting up Node.js dependencies..."
    
    cd frontend
    
    if [ ! -d "node_modules" ]; then
        npm install
        print_success "Installed Node.js dependencies"
    else
        print_success "Node.js dependencies already installed"
    fi
    
    cd ..
}

# Create development environment file
setup_dev_environment() {
    print_status "Setting up development environment..."
    
    if [ ! -f .env ]; then
        if [ -f env.example ]; then
            cp env.example .env
            print_success "Created .env file from env.example"
        else
            print_error "env.example file not found"
            exit 1
        fi
    else
        print_success ".env file already exists"
    fi
    
    # Update .env for development
    sed -i.bak 's/localhost/127.0.0.1/g' .env
    print_success "Updated .env for local development"
}

# Install system dependencies (optional)
install_system_deps() {
    print_status "Checking system dependencies..."
    
    # Check if PostgreSQL is installed
    if ! command -v psql &> /dev/null; then
        print_warning "PostgreSQL is not installed. You'll need it for local development."
        print_warning "Install with: brew install postgresql (macOS) or apt-get install postgresql (Ubuntu)"
    else
        print_success "PostgreSQL is installed"
    fi
    
    # Check if Redis is installed
    if ! command -v redis-server &> /dev/null; then
        print_warning "Redis is not installed. You'll need it for local development."
        print_warning "Install with: brew install redis (macOS) or apt-get install redis (Ubuntu)"
    else
        print_success "Redis is installed"
    fi
}

# Create development scripts
create_dev_scripts() {
    print_status "Creating development scripts..."
    
    # Backend start script
    cat > start-backend.sh << 'EOF'
#!/bin/bash
source venv/bin/activate
export PYTHONPATH=$PWD:$PYTHONPATH
python main.py --mode api --host 127.0.0.1 --port 8000
EOF
    chmod +x start-backend.sh
    
    # Frontend start script
    cat > start-frontend.sh << 'EOF'
#!/bin/bash
cd frontend
npm start
EOF
    chmod +x start-frontend.sh
    
    # Database setup script
    cat > setup-db.sh << 'EOF'
#!/bin/bash
# Start PostgreSQL and Redis for local development
echo "Starting PostgreSQL and Redis services..."

# Start PostgreSQL (adjust based on your system)
if command -v brew &> /dev/null; then
    brew services start postgresql
elif command -v systemctl &> /dev/null; then
    sudo systemctl start postgresql
fi

# Start Redis
if command -v brew &> /dev/null; then
    brew services start redis
elif command -v systemctl &> /dev/null; then
    sudo systemctl start redis
fi

echo "Database services started"
EOF
    chmod +x setup-db.sh
    
    print_success "Development scripts created"
}

# Show development setup status
show_dev_status() {
    print_success "🎉 Development environment setup completed!"
    echo ""
    echo "📋 Next Steps:"
    echo "   1. Start database services: ./setup-db.sh"
    echo "   2. Start backend: ./start-backend.sh"
    echo "   3. Start frontend: ./start-frontend.sh"
    echo ""
    echo "🔧 Available Scripts:"
    echo "   ./start-backend.sh  - Start the backend API server"
    echo "   ./start-frontend.sh - Start the frontend development server"
    echo "   ./setup-db.sh       - Start PostgreSQL and Redis services"
    echo ""
    echo "📊 Service URLs (when running):"
    echo "   Frontend:     http://localhost:3000"
    echo "   Backend API:  http://localhost:8000"
    echo "   API Docs:     http://localhost:8000/docs"
    echo ""
    echo "💡 Tips:"
    echo "   - Use 'source venv/bin/activate' to activate Python environment"
    echo "   - Check .env file for configuration options"
    echo "   - Use 'docker-compose up' for full Docker setup"
}

# Main setup function
main() {
    print_status "Starting development environment setup..."
    
    check_python
    check_node
    setup_python_env
    setup_node_env
    setup_dev_environment
    install_system_deps
    create_dev_scripts
    show_dev_status
}

# Handle script arguments
case "${1:-}" in
    "clean")
        print_status "Cleaning development environment..."
        rm -rf venv
        rm -rf frontend/node_modules
        rm -f .env
        rm -f start-*.sh setup-db.sh
        print_success "Development environment cleaned"
        ;;
    *)
        main
        ;;
esac
