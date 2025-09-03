#!/usr/bin/env python3
"""
KGAREVION Medical QA System - Main Application Entry Point
Combines all modules for full-stack deployment
"""

import os
import sys
import asyncio
import logging
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Import our modules
from kgarevion_core import KGARevionPipeline, Question
from kgarevion_api import app
import uvicorn

async def initialize_system():
    """Initialize all system components"""
    logger.info("Initializing KGAREVION Medical QA System...")
    
    # Check for required environment variables
    required_vars = [
        'NEO4J_URI',
        'NEO4J_USER', 
        'NEO4J_PASSWORD',
        'REDIS_HOST',
        'DATABASE_URL'
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.warning(f"Missing environment variables: {missing_vars}")
        logger.info("Using default values for development...")
        
        # Set defaults for development
        os.environ.setdefault('NEO4J_URI', 'neo4j://localhost:7687')
        os.environ.setdefault('NEO4J_USER', 'neo4j')
        os.environ.setdefault('NEO4J_PASSWORD', 'password')
        os.environ.setdefault('REDIS_HOST', 'localhost')
        os.environ.setdefault('DATABASE_URL', 'postgresql://medical_qa:password@localhost:5432/kgarevion')
    
    logger.info("System initialization complete")
    return True

def run_api_server():
    """Run the FastAPI server"""
    logger.info("Starting FastAPI server...")
    uvicorn.run(
        "kgarevion_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

async def run_demo():
    """Run a demo of the system"""
    logger.info("Running KGAREVION demo...")
    
    # Initialize pipeline
    pipeline = KGARevionPipeline(
        llm_model="meta-llama/Llama-3-8B-Instruct",
        kg_uri=os.getenv('NEO4J_URI', 'neo4j://localhost:7687'),
        kg_user=os.getenv('NEO4J_USER', 'neo4j'),
        kg_password=os.getenv('NEO4J_PASSWORD', 'password'),
        redis_host=os.getenv('REDIS_HOST', 'localhost')
    )
    
    # Demo questions
    demo_questions = [
        Question(
            text="Which protein is associated with Retinitis Pigmentosa 59?",
            question_type="multiple_choice",
            candidates=["HSPA4", "HSPA8", "HSPA1B", "HSPA1A"]
        ),
        Question(
            text="What are the contraindications for aspirin?",
            question_type="open_ended"
        ),
        Question(
            text="Which gene mutations are linked to cystic fibrosis?",
            question_type="open_ended"
        )
    ]
    
    for i, question in enumerate(demo_questions, 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Demo Question {i}: {question.text}")
        logger.info(f"{'='*60}")
        
        try:
            result = await pipeline.process_question(question)
            logger.info(f"Answer: {result['answer']}")
            logger.info(f"Confidence: {result.get('confidence', 'N/A')}")
            logger.info(f"Verified Triplets: {len(result.get('verified_triplets', []))}")
            logger.info(f"Medical Entities: {result.get('medical_entities', [])}")
        except Exception as e:
            logger.error(f"Error processing question: {e}")
    
    await pipeline.close()
    logger.info("Demo complete!")

def main():
    """Main entry point"""
    import argparse
    
    parser = argparse.ArgumentParser(description='KGAREVION Medical QA System')
    parser.add_argument('--mode', choices=['api', 'demo', 'init'], default='api',
                      help='Run mode: api (start API server), demo (run demo), init (initialize only)')
    parser.add_argument('--host', default='0.0.0.0', help='API server host')
    parser.add_argument('--port', type=int, default=8000, help='API server port')
    
    args = parser.parse_args()
    
    # Initialize system
    asyncio.run(initialize_system())
    
    if args.mode == 'api':
        run_api_server()
    elif args.mode == 'demo':
        asyncio.run(run_demo())
    elif args.mode == 'init':
        logger.info("System initialized. Ready for deployment.")
    else:
        logger.error(f"Unknown mode: {args.mode}")
        sys.exit(1)

if __name__ == "__main__":
    main()