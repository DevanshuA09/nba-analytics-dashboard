#!/usr/bin/env python3
"""
Startup script for NBA Analytics Dashboard
Starts the FastAPI server with proper configuration
"""

import uvicorn
import os
import sys
from pathlib import Path

def main():
    """Start the FastAPI server"""
    
    # Check if models exist
    models_dir = Path("models")
    team_model = models_dir / "team_win.pkl"
    player_model = models_dir / "player_points.pkl"
    
    if not team_model.exists() or not player_model.exists():
        print("⚠️  Models not found. Please train the models first:")
        print("   python train_models.py")
        print()
        print("Starting server anyway (predictions will not work without models)...")
    
    # Get port from environment or use default
    port = int(os.environ.get("PORT", 8000))
    host = os.environ.get("HOST", "0.0.0.0")
    
    print(f"🚀 Starting NBA Analytics Dashboard API on {host}:{port}")
    print(f"📊 API Documentation: http://{host}:{port}/docs")
    print(f"🏀 Health Check: http://{host}:{port}/health")
    print()
    print("Press Ctrl+C to stop the server")
    
    # Start the server
    uvicorn.run(
        "app:app",
        host=host,
        port=port,
        reload=True,
        log_level="info"
    )

if __name__ == "__main__":
    main()
