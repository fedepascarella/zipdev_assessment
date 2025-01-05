import uvicorn
import subprocess
import threading
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define the services to be started
services = [
    {"name": "upload_files_service", "module": "upload_files_service", "port": 8002},
    {"name": "indexing_service", "module": "indexing_service", "port": 8000},
    {"name": "intent_extraction_service", "module": "intent_extraction_service", "port": 8003},
    {"name": "search_service", "module": "search_service", "port": 8004},
    {"name": "ai_models_service", "module": "ai_models_service", "port": 8005},
    {"name": "response_service", "module": "response_service", "port": 8006},
    {"name": "index_excel_service", "module": "indexing_service", "port": 8000},
]


def run_service(module: str, port: int):
    """Run a service as a subprocess using uvicorn."""
    try:
        subprocess.run([
            "uvicorn", f"{module}:app", "--host", "0.0.0.0", f"--port", str(port)
        ])
    except Exception as e:
        print(f"Failed to start service {module} on port {port}: {e}")


def start_services():
    """Start all services in separate threads."""
    threads = []
    for service in services:
        thread = threading.Thread(target=run_service, args=(service["module"], service["port"]))
        thread.daemon = True
        thread.start()
        threads.append(thread)
        print(f"Started {service['name']} on port {service['port']}")

    # Keep main thread running to avoid early exit
    for thread in threads:
        thread.join()


def start_frontend():
    """Run the frontend service using Streamlit."""
    try:
        subprocess.run(["streamlit", "run", "frontend_service.py"])
    except Exception as e:
        print(f"Failed to start the frontend: {e}")


if __name__ == "__main__":
    try:
        # Start backend services in a separate thread
        backend_thread = threading.Thread(target=start_services)
        backend_thread.start()

        # Start the frontend service in the main thread
        start_frontend()
    except KeyboardInterrupt:
        print("Shutting down services...")
