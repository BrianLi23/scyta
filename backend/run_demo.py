#!/usr/bin/env python3
"""
SCYTA Demo Launcher
==================

Quick launcher for the SCYTA multi-agent system demo.
This script ensures the environment is set up correctly before starting the demo.
"""

import os
import sys
import subprocess
from pathlib import Path
from dotenv import load_dotenv

def check_dependencies():
    """Check if required dependencies are installed."""
    required_packages = [
        'groq',
        'pathlib', 
        'typing',
        'json',
        'datetime'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing required packages: {', '.join(missing)}")
        print("Please install them using: pip install -r requirements.txt")
        return False
    
    return True

def check_environment():
    """Check if environment variables are set."""
    required_env = ['GROQ_API']
    missing = []
    
    for env_var in required_env:
        if not os.getenv(env_var):
            missing.append(env_var)
    
    if missing:
        print(f"❌ Missing environment variables: {', '.join(missing)}")
        print("Please set them in your .env file or environment.")
        return False
        
    return True

def main():
    """Main launcher function."""
    print("🚀 SCYTA Demo Launcher")
    print("=" * 30)
    
    # Set project directory to the backend folder
    project_root = Path(__file__).resolve().parent
    os.chdir(project_root)
    
    print(f"📁 Project directory: {project_root}")
    
    load_dotenv(dotenv_path=project_root / '.env')
    
    # Check dependencies
    print("🔍 Checking dependencies...")
    if not check_dependencies():
        sys.exit(1)
        
    # Check environment
    print("🔍 Checking environment...")
    if not check_environment():
        print("💡 Tip: Create a .env file with your GROQ_API key")
        sys.exit(1)
    
    print("✅ All checks passed!")
    print("🎯 Starting SCYTA demo...\n")
    
    # Import and run the demo
    try:
        from demo import main as demo_main
        demo_main()
    except ImportError as e:
        print(f"❌ Failed to import demo: {e}")
        print("Please ensure you're in the correct directory and all files are present.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Failed to start demo: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
