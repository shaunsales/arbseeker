#!/usr/bin/env python3
"""
Run the Strategy Lab web application.

Usage:
    python run_app.py              # Run on port 8000
    python run_app.py --port 8080  # Custom port
"""

import argparse
import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run Strategy Lab web app")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    
    args = parser.parse_args()
    
    print(f"\n🚀 Starting Strategy Lab at http://{args.host}:{args.port}\n")
    
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
