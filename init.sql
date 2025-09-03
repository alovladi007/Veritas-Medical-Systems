-- KGAREVION Medical QA System Database Schema

CREATE SCHEMA IF NOT EXISTS kgarevion;

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    username VARCHAR(255) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    role VARCHAR(50) DEFAULT 'user',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- QA Logs table
CREATE TABLE IF NOT EXISTS qa_logs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID REFERENCES users(id),
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    confidence_score FLOAT,
    processing_time_ms INTEGER,
    triplets JSONB,
    medical_entities TEXT[],
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Model versions table
CREATE TABLE IF NOT EXISTS model_versions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    model_name VARCHAR(255) NOT NULL,
    version VARCHAR(50) NOT NULL,
    accuracy FLOAT,
    parameters JSONB,
    is_active BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Triplet cache table
CREATE TABLE IF NOT EXISTS triplet_cache (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    question_hash VARCHAR(64) NOT NULL,
    triplets JSONB NOT NULL,
    verified BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

-- Create indexes for better performance
CREATE INDEX idx_qa_logs_user_id ON qa_logs(user_id);
CREATE INDEX idx_qa_logs_created_at ON qa_logs(created_at);
CREATE INDEX idx_triplet_cache_question_hash ON triplet_cache(question_hash);
CREATE INDEX idx_triplet_cache_expires ON triplet_cache(expires_at);

-- Insert default admin user (password: admin123)
INSERT INTO users (username, email, password_hash, role) 
VALUES ('admin', 'admin@kgarevion.com', '$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/LewY.JeJlyhJlKNyi', 'admin')
ON CONFLICT (username) DO NOTHING;