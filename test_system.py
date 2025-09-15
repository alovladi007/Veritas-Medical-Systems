#!/usr/bin/env python3
"""
KGAREVION Medical QA System Test Script
Tests the core functionality of the system
"""

import asyncio
import sys
import os
import json
from pathlib import Path

# Add the current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

async def test_core_pipeline():
    """Test the core KGAREVION pipeline"""
    print("🧪 Testing Core Pipeline...")
    
    try:
        from kgarevion_core import KGARevionPipeline, Question
        
        # Initialize pipeline (this will fail without proper setup, but we can test the structure)
        print("✅ Core pipeline imports successful")
        
        # Test Question creation
        question = Question(
            text="Which protein is associated with Retinitis Pigmentosa 59?",
            question_type="multiple_choice",
            candidates=["HSPA4", "HSPA8", "HSPA1B", "HSPA1A"]
        )
        print("✅ Question object creation successful")
        
        return True
    except Exception as e:
        print(f"❌ Core pipeline test failed: {e}")
        return False

def test_api_structure():
    """Test API structure and imports"""
    print("🧪 Testing API Structure...")
    
    try:
        from kgarevion_api import app
        print("✅ FastAPI app import successful")
        
        # Check if main endpoints exist
        routes = [route.path for route in app.routes]
        expected_routes = [
            "/health",
            "/api/medical-qa",
            "/api/medical-qa/stream",
            "/ws/medical-qa",
            "/api/auth/register",
            "/api/auth/login"
        ]
        
        for route in expected_routes:
            if any(route in r for r in routes):
                print(f"✅ Route {route} found")
            else:
                print(f"⚠️  Route {route} not found")
        
        return True
    except Exception as e:
        print(f"❌ API structure test failed: {e}")
        return False

def test_alignment_module():
    """Test the alignment module"""
    print("🧪 Testing Alignment Module...")
    
    try:
        from kgarevion_alignment import KGtoLLMAlignmentModule, RELATION_DESCRIPTIONS
        
        print("✅ Alignment module imports successful")
        print(f"✅ Found {len(RELATION_DESCRIPTIONS)} relation descriptions")
        
        return True
    except Exception as e:
        print(f"❌ Alignment module test failed: {e}")
        return False

def test_dependencies():
    """Test if all required dependencies are available"""
    print("🧪 Testing Dependencies...")
    
    required_packages = [
        'fastapi',
        'uvicorn',
        'transformers',
        'torch',
        'neo4j',
        'redis',
        'pydantic',
        'numpy',
        'sentence_transformers'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - Missing")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
        print("Install with: pip install -r requirements.txt")
        return False
    
    return True

def test_environment():
    """Test environment setup"""
    print("🧪 Testing Environment...")
    
    # Check if .env file exists
    env_file = Path('.env')
    env_example = Path('env.example')
    
    if env_file.exists():
        print("✅ .env file exists")
    elif env_example.exists():
        print("⚠️  .env file not found, but env.example exists")
        print("   Copy env.example to .env and configure it")
    else:
        print("❌ No environment files found")
        return False
    
    # Check Docker Compose file
    docker_compose = Path('docker-compose.yml')
    if docker_compose.exists():
        print("✅ docker-compose.yml exists")
    else:
        print("❌ docker-compose.yml not found")
        return False
    
    # Check deployment scripts
    deploy_script = Path('deploy.sh')
    setup_script = Path('setup-dev.sh')
    
    if deploy_script.exists() and deploy_script.stat().st_mode & 0o111:
        print("✅ deploy.sh exists and is executable")
    else:
        print("❌ deploy.sh not found or not executable")
        return False
    
    if setup_script.exists() and setup_script.stat().st_mode & 0o111:
        print("✅ setup-dev.sh exists and is executable")
    else:
        print("❌ setup-dev.sh not found or not executable")
        return False
    
    return True

def test_frontend_structure():
    """Test frontend structure"""
    print("🧪 Testing Frontend Structure...")
    
    frontend_path = Path('frontend')
    if not frontend_path.exists():
        print("❌ Frontend directory not found")
        return False
    
    # Check package.json
    package_json = frontend_path / 'package.json'
    if package_json.exists():
        print("✅ package.json exists")
    else:
        print("❌ package.json not found")
        return False
    
    # Check main React files
    app_tsx = frontend_path / 'src' / 'App.tsx'
    index_tsx = frontend_path / 'src' / 'index.tsx'
    
    if app_tsx.exists():
        print("✅ App.tsx exists")
    else:
        print("❌ App.tsx not found")
        return False
    
    if index_tsx.exists():
        print("✅ index.tsx exists")
    else:
        print("❌ index.tsx not found")
        return False
    
    # Check UI components
    ui_components = [
        'card.tsx',
        'alert.tsx',
        'button.tsx',
        'input.tsx',
        'textarea.tsx'
    ]
    
    components_path = frontend_path / 'src' / 'components' / 'ui'
    for component in ui_components:
        component_file = components_path / component
        if component_file.exists():
            print(f"✅ {component} exists")
        else:
            print(f"⚠️  {component} not found")
    
    return True

async def main():
    """Run all tests"""
    print("🚀 KGAREVION Medical QA System - System Test")
    print("=" * 50)
    
    tests = [
        ("Dependencies", test_dependencies),
        ("Environment", test_environment),
        ("Frontend Structure", test_frontend_structure),
        ("API Structure", test_api_structure),
        ("Alignment Module", test_alignment_module),
        ("Core Pipeline", test_core_pipeline),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        print("-" * 20)
        
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:<20} {status}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! System is ready for deployment.")
        return 0
    else:
        print("⚠️  Some tests failed. Please fix the issues before deployment.")
        return 1

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
