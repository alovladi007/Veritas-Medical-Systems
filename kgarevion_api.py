"""
FastAPI Backend for KGAREVION Medical QA System
"""

from fastapi import FastAPI, HTTPException, WebSocket, BackgroundTasks, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import asyncio
import json
import uuid
from datetime import datetime
import asyncpg
from redis import asyncio as redis
import logging
from contextlib import asynccontextmanager
import jwt
from passlib.context import CryptContext

# Import our core pipeline
# from kgarevion_core import KGARevionPipeline, Question

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Security
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
SECRET_KEY = "your-secret-key-change-in-production"
ALGORITHM = "HS256"


# Pydantic models
class QuestionRequest(BaseModel):
    text: str = Field(..., description="The medical question text")
    question_type: str = Field("multiple_choice", description="Type: multiple_choice or open_ended")
    candidates: Optional[List[str]] = Field(None, description="Answer candidates for MC questions")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")


class QuestionResponse(BaseModel):
    question_id: str
    question: str
    answer: str
    confidence_score: float
    verified_triplets: List[Dict[str, str]]
    medical_entities: List[str]
    processing_time_ms: int
    explanation: Optional[str] = None


class TripletVisualization(BaseModel):
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]


class UserCreate(BaseModel):
    username: str
    email: str
    password: str
    role: str = "user"


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class HealthCheck(BaseModel):
    status: str
    timestamp: datetime
    services: Dict[str, bool]


# Database manager
class DatabaseManager:
    def __init__(self):
        self.pool = None
        
    async def initialize(self):
        self.pool = await asyncpg.create_pool(
            host="localhost",
            port=5432,
            user="medical_qa",
            password="password",
            database="kgarevion",
            min_size=10,
            max_size=20
        )
        
    async def close(self):
        if self.pool:
            await self.pool.close()
            
    async def log_question(self, user_id: str, question: str, answer: str, processing_time: int):
        async with self.pool.acquire() as conn:
            await conn.execute("""
                INSERT INTO qa_logs (user_id, question, answer, processing_time_ms, created_at)
                VALUES ($1, $2, $3, $4, $5)
            """, user_id, question, answer, processing_time, datetime.utcnow())
            
    async def get_user_history(self, user_id: str, limit: int = 10):
        async with self.pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT question, answer, created_at, processing_time_ms
                FROM qa_logs
                WHERE user_id = $1
                ORDER BY created_at DESC
                LIMIT $2
            """, user_id, limit)
            return [dict(row) for row in rows]


# Application lifespan manager
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting KGAREVION Medical QA Service...")
    
    # Initialize database
    app.state.db = DatabaseManager()
    await app.state.db.initialize()
    
    # Initialize Redis
    app.state.redis = redis.Redis(host="localhost", port=6379, decode_responses=True)
    
    # Initialize KG pipeline
    # app.state.pipeline = KGARevionPipeline(
    #     llm_model="meta-llama/Llama-3-8B-Instruct",
    #     kg_uri="neo4j://localhost:7687",
    #     kg_user="neo4j",
    #     kg_password="password"
    # )
    
    # Create ML model cache
    app.state.model_cache = {}
    
    yield
    
    # Shutdown
    logger.info("Shutting down KGAREVION Medical QA Service...")
    await app.state.db.close()
    await app.state.redis.close()
    # await app.state.pipeline.close()


# Create FastAPI app
app = FastAPI(
    title="KGAREVION Medical QA API",
    description="Knowledge Graph-based Medical Question Answering System",
    version="1.0.0",
    lifespan=lifespan
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Authentication dependency
async def get_current_user(request: Request):
    """Extract and verify JWT token"""
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        return None
    
    token = auth_header.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload.get("sub")  # user_id
    except jwt.PyJWTError:
        return None


# Health check endpoint
@app.get("/health", response_model=HealthCheck)
async def health_check():
    """Check system health status"""
    services_status = {
        "database": False,
        "redis": False,
        "knowledge_graph": False,
        "llm_service": False
    }
    
    # Check database
    try:
        async with app.state.db.pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
            services_status["database"] = True
    except:
        pass
    
    # Check Redis
    try:
        await app.state.redis.ping()
        services_status["redis"] = True
    except:
        pass
    
    # Check other services...
    
    all_healthy = all(services_status.values())
    
    return HealthCheck(
        status="healthy" if all_healthy else "degraded",
        timestamp=datetime.utcnow(),
        services=services_status
    )


# Main QA endpoint
@app.post("/api/medical-qa", response_model=QuestionResponse)
async def process_medical_question(
    request: QuestionRequest,
    background_tasks: BackgroundTasks,
    user_id: Optional[str] = Depends(get_current_user)
):
    """Process a medical question through KGAREVION pipeline"""
    
    start_time = asyncio.get_event_loop().time()
    question_id = str(uuid.uuid4())
    
    try:
        # Check rate limiting
        if user_id:
            rate_key = f"rate:{user_id}"
            count = await app.state.redis.incr(rate_key)
            if count == 1:
                await app.state.redis.expire(rate_key, 60)
            elif count > 10:  # 10 requests per minute
                raise HTTPException(status_code=429, detail="Rate limit exceeded")
        
        # Create Question object
        # question = Question(
        #     text=request.text,
        #     question_type=request.question_type,
        #     candidates=request.candidates
        # )
        
        # Process through pipeline
        # result = await app.state.pipeline.process_question(question)
        
        # Mock result for demonstration
        result = {
            "answer": "B: HSPA8",
            "verified_triplets": [
                {"head": "HSPA8", "relation": "interacts_with", "tail": "DHDDS"},
                {"head": "DHDDS", "relation": "associated_with", "tail": "Retinitis Pigmentosa 59"}
            ],
            "medical_entities": ["HSPA8", "DHDDS", "Retinitis Pigmentosa"],
            "confidence": 0.92
        }
        
        processing_time = int((asyncio.get_event_loop().time() - start_time) * 1000)
        
        # Log to database in background
        if user_id:
            background_tasks.add_task(
                app.state.db.log_question,
                user_id,
                request.text,
                result["answer"],
                processing_time
            )
        
        return QuestionResponse(
            question_id=question_id,
            question=request.text,
            answer=result["answer"],
            confidence_score=result.get("confidence", 0.0),
            verified_triplets=result["verified_triplets"],
            medical_entities=result["medical_entities"],
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error processing question {question_id}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


# Streaming endpoint for real-time responses
@app.post("/api/medical-qa/stream")
async def stream_medical_question(request: QuestionRequest):
    """Stream processing updates in real-time"""
    
    async def generate():
        # Stream processing stages
        stages = [
            {"stage": "extracting_entities", "progress": 0.25},
            {"stage": "generating_triplets", "progress": 0.5},
            {"stage": "reviewing_triplets", "progress": 0.75},
            {"stage": "generating_answer", "progress": 1.0}
        ]
        
        for stage in stages:
            yield f"data: {json.dumps(stage)}\n\n"
            await asyncio.sleep(1)  # Simulate processing time
            
        # Final result
        result = {
            "stage": "complete",
            "answer": "The answer based on verified knowledge...",
            "triplets": []
        }
        yield f"data: {json.dumps(result)}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


# WebSocket for real-time interaction
@app.websocket("/ws/medical-qa")
async def websocket_medical_qa(websocket: WebSocket):
    """WebSocket endpoint for real-time Q&A"""
    await websocket.accept()
    
    try:
        while True:
            # Receive question
            data = await websocket.receive_json()
            
            # Send processing updates
            await websocket.send_json({
                "type": "status",
                "message": "Processing your question..."
            })
            
            # Process question (simplified)
            await asyncio.sleep(2)
            
            # Send triplets as they're generated
            await websocket.send_json({
                "type": "triplet",
                "data": {
                    "head": "HSPA8",
                    "relation": "interacts_with",
                    "tail": "DHDDS"
                }
            })
            
            # Send final answer
            await websocket.send_json({
                "type": "answer",
                "data": {
                    "text": "Based on the analysis, the answer is HSPA8",
                    "confidence": 0.92
                }
            })
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        await websocket.close()


# Triplet visualization endpoint
@app.get("/api/visualization/{question_id}", response_model=TripletVisualization)
async def get_triplet_visualization(question_id: str):
    """Get visualization data for triplets"""
    
    # Fetch triplets from cache
    cache_key = f"viz:{question_id}"
    cached = await app.state.redis.get(cache_key)
    
    if cached:
        return json.loads(cached)
    
    # Generate visualization data
    nodes = [
        {"id": "HSPA8", "label": "HSPA8", "type": "protein", "color": "#4CAF50"},
        {"id": "DHDDS", "label": "DHDDS", "type": "gene", "color": "#2196F3"},
        {"id": "RP59", "label": "Retinitis Pigmentosa 59", "type": "disease", "color": "#FF5722"}
    ]
    
    edges = [
        {"source": "HSPA8", "target": "DHDDS", "label": "interacts_with", "weight": 0.9},
        {"source": "DHDDS", "target": "RP59", "label": "associated_with", "weight": 0.85}
    ]
    
    viz_data = TripletVisualization(nodes=nodes, edges=edges)
    
    # Cache visualization
    await app.state.redis.setex(cache_key, 3600, json.dumps(viz_data.dict()))
    
    return viz_data


# User history endpoint
@app.get("/api/history")
async def get_user_history(
    limit: int = 10,
    user_id: Optional[str] = Depends(get_current_user)
):
    """Get user's question history"""
    
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    
    history = await app.state.db.get_user_history(user_id, limit)
    return {"history": history}


# Admin endpoints
@app.post("/api/admin/kg/update")
async def update_knowledge_graph(
    triplets: List[Dict[str, str]],
    user_id: Optional[str] = Depends(get_current_user)
):
    """Update knowledge graph with new triplets (admin only)"""
    
    # Check admin privileges
    # if not is_admin(user_id):
    #     raise HTTPException(status_code=403, detail="Admin access required")
    
    # Update KG
    # for triplet in triplets:
    #     await app.state.pipeline.kg.add_triplet(triplet)
    
    return {"message": f"Added {len(triplets)} triplets to knowledge graph"}


@app.post("/api/admin/model/fine-tune")
async def trigger_fine_tuning(
    background_tasks: BackgroundTasks,
    user_id: Optional[str] = Depends(get_current_user)
):
    """Trigger model fine-tuning (admin only)"""
    
    # Check admin privileges
    # if not is_admin(user_id):
    #     raise HTTPException(status_code=403, detail="Admin access required")
    
    # Trigger fine-tuning in background
    # background_tasks.add_task(fine_tune_model)
    
    return {"message": "Fine-tuning job started", "job_id": str(uuid.uuid4())}


# Metrics endpoint
@app.get("/api/metrics")
async def get_system_metrics():
    """Get system performance metrics"""
    
    metrics = {
        "total_questions_processed": await app.state.redis.get("metrics:total_questions") or 0,
        "average_processing_time_ms": await app.state.redis.get("metrics:avg_time") or 0,
        "cache_hit_rate": await app.state.redis.get("metrics:cache_hit_rate") or 0,
        "active_users": await app.state.redis.scard("active_users") or 0,
        "triplets_in_kg": 4050249,  # From PrimeKG
        "model_accuracy": 0.734  # From paper results
    }
    
    return metrics


# Authentication endpoints
@app.post("/api/auth/register", response_model=Token)
async def register_user(user: UserCreate):
    """Register a new user"""
    
    # Hash password
    hashed_password = pwd_context.hash(user.password)
    
    # Save to database
    async with app.state.db.pool.acquire() as conn:
        user_id = await conn.fetchval("""
            INSERT INTO users (username, email, password_hash, role, created_at)
            VALUES ($1, $2, $3, $4, $5)
            RETURNING id
        """, user.username, user.email, hashed_password, user.role, datetime.utcnow())
    
    # Generate JWT token
    token = jwt.encode(
        {"sub": str(user_id), "exp": datetime.utcnow().timestamp() + 86400},
        SECRET_KEY,
        algorithm=ALGORITHM
    )
    
    return Token(access_token=token)


@app.post("/api/auth/login", response_model=Token)
async def login(username: str, password: str):
    """Authenticate user and return token"""
    
    # Verify credentials
    async with app.state.db.pool.acquire() as conn:
        user = await conn.fetchrow("""
            SELECT id, password_hash FROM users WHERE username = $1
        """, username)
        
    if not user or not pwd_context.verify(password, user["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    
    # Generate JWT token
    token = jwt.encode(
        {"sub": str(user["id"]), "exp": datetime.utcnow().timestamp() + 86400},
        SECRET_KEY,
        algorithm=ALGORITHM
    )
    
    return Token(access_token=token)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
