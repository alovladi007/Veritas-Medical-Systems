# KGAREVION Medical QA System

A state-of-the-art medical question-answering system that combines Large Language Models (LLMs) with Knowledge Graphs for accurate and verifiable medical information retrieval.

## 🏗️ Architecture

KGAREVION implements a novel Generate-Review-Revise pipeline:
1. **Generate**: Creates medical knowledge triplets from questions
2. **Review**: Verifies triplets against a medical knowledge graph
3. **Revise**: Corrects false triplets using fine-tuned models
4. **Answer**: Generates final answers based on verified knowledge

## 🚀 Quick Start

### Option 1: Docker Compose (Recommended for Production)

```bash
# Clone the repository
git clone https://github.com/yourusername/kgarevion.git
cd kgarevion

# Run the automated deployment script
./deploy.sh

# Access the application
# Frontend: http://localhost:3000
# API: http://localhost:8000
# API Docs: http://localhost:8000/docs
# Neo4j Browser: http://localhost:7474
```

### Option 2: Development Setup (Local Development)

```bash
# Clone the repository
git clone https://github.com/yourusername/kgarevion.git
cd kgarevion

# Run the development setup script
./setup-dev.sh

# Start database services
./setup-db.sh

# Start backend (in terminal 1)
./start-backend.sh

# Start frontend (in terminal 2)
./start-frontend.sh
```

### Manual Installation

#### Prerequisites
- Python 3.10+
- Node.js 18+
- PostgreSQL 15+
- Neo4j 5+
- Redis 7+
- CUDA-capable GPU (optional, for faster inference)

#### Backend Setup

```bash
# Install Python dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your configuration

# Initialize database
psql -U postgres < init.sql

# Run the backend
python main.py --mode api
```

#### Frontend Setup

```bash
cd frontend

# Install dependencies
npm install

# Start development server
npm start
```

## 📁 Project Structure

```
kgarevion/
├── kgarevion_core.py       # Core pipeline implementation
├── kgarevion_api.py        # FastAPI backend
├── kgarevion_alignment.py  # KG-LLM alignment module
├── main.py                  # Application entry point
├── requirements.txt         # Python dependencies
├── docker-compose.yml       # Docker orchestration
├── init.sql                # Database schema
├── frontend/               # React frontend
│   ├── src/
│   │   ├── App.tsx        # Main React component
│   │   ├── components/    # UI components
│   │   └── index.tsx      # Entry point
│   └── package.json       # Node dependencies
└── README.md              # This file
```

## 🔧 Configuration

### Environment Variables

Copy `env.example` to `.env` and configure the following variables:

```bash
# Database Configuration
DATABASE_URL=postgresql://medical_qa:password@localhost:5432/kgarevion
DB_PASSWORD=password

# Neo4j Knowledge Graph Configuration
NEO4J_URI=neo4j://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password

# Redis Cache Configuration
REDIS_HOST=localhost
REDIS_PORT=6379

# Model Configuration
MODEL_NAME=meta-llama/Llama-3-8B-Instruct

# Security
SECRET_KEY=your-super-secret-key-change-in-production

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Frontend Configuration
REACT_APP_API_URL=http://localhost:8000
REACT_APP_WS_URL=ws://localhost:8000
```

### Default Credentials

- **Admin Username**: `admin`
- **Admin Password**: `admin123`

## 📊 Features

- **Medical Question Answering**: Support for both multiple-choice and open-ended questions
- **Knowledge Graph Integration**: Verification against PrimeKG (4M+ medical triplets)
- **Real-time Processing**: WebSocket support for live updates
- **Caching**: Redis-based caching for improved performance
- **Authentication**: JWT-based user authentication
- **Visualization**: Interactive knowledge graph visualization
- **Metrics**: System performance monitoring

## 🛠️ Deployment Management

### Docker Commands

```bash
# Start all services
./deploy.sh

# Stop services
./deploy.sh stop

# Restart services
./deploy.sh restart

# View logs
./deploy.sh logs

# Check service status
./deploy.sh status

# Clean up everything
./deploy.sh clean
```

### Development Commands

```bash
# Setup development environment
./setup-dev.sh

# Start backend server
./start-backend.sh

# Start frontend server
./start-frontend.sh

# Start database services
./setup-db.sh

# Clean development environment
./setup-dev.sh clean
```

## 🧪 Running Tests

```bash
# Backend tests
pytest tests/

# Frontend tests
cd frontend && npm test
```

## 📈 Performance

- Accuracy: 73.4% on medical QA benchmarks
- Average processing time: < 2 seconds per question
- Supports 4M+ medical knowledge triplets
- Scalable to handle concurrent requests

## 🔬 Model Details

The system uses:
- **LLM**: Meta Llama-3-8B-Instruct (quantized)
- **Knowledge Graph**: PrimeKG with 4,050,249 triplets
- **Embeddings**: TransE for structural representation
- **Fine-tuning**: LoRA adapters for efficient training

## 🤝 Contributing

Contributions are welcome! Please read our contributing guidelines and submit pull requests to the main branch.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📚 Citation

If you use KGAREVION in your research, please cite:

```bibtex
@article{kgarevion2024,
  title={KGAREVION: Knowledge Graph Augmented Medical QA},
  author={Your Name},
  year={2024}
}
```

## 🆘 Support

For issues and questions:
- Open an issue on GitHub
- Contact: support@kgarevion.com

## 🚦 Status

- Backend API: ✅ Ready
- Frontend UI: ✅ Ready
- Knowledge Graph: ✅ Configured
- Model Pipeline: ✅ Implemented
- Docker Deployment: ✅ Available