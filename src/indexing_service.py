from fastapi import FastAPI, HTTPException, UploadFile, File
import os
import json
import sqlite3
import logging
import requests
import uuid
from typing import List
from pydantic import BaseModel
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv
from PyPDF2 import PdfReader
import shutil
import uvicorn

# Load environment variables
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
MODELS_SERVICE_URL = os.getenv("MODELS_SERVICE_URL", "http://localhost:8005")
UPLOAD_DIR = os.path.join("src", "main", "docs")
INDEXED_FILES_PATH = "./src/main/docs"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Initialize FastAPI app
app = FastAPI(title="Indexing Service", description="Indexing Service for generating and storing embeddings from raw data")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# SQLite Database setup
db_path = "./db/indexing_service_logs.db"

# In-memory cache to store indexed files
indexed_files_cache = set()

# Qdrant client initialization
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

# Check if Qdrant collection exists
def ensure_collection_exists(collection_name: str, vector_size: int = 768):
    collections = qdrant_client.get_collections().collections
    if not any(col.name == collection_name for col in collections):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=vector_size, distance=models.Distance.COSINE)
        )
        logging.info(f"Created new collection: {collection_name}")
    else:
        logging.info(f"Collection '{collection_name}' already exists.")

# Create table if it doesn't exist
def init_db():
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS logs (
               id INTEGER PRIMARY KEY AUTOINCREMENT,
               endpoint TEXT NOT NULL,
               request_body TEXT NOT NULL,
               response_body TEXT,
               timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
           )"""
    )
    conn.commit()
    conn.close()

# Call database initialization
init_db()

# Pydantic models for request and response
class EmbeddingRequest(BaseModel):
    texts: List[str]

class EmbeddingResponse(BaseModel):
    message: str

@app.post("/index", response_model=EmbeddingResponse)
async def generate_and_store_embeddings(request: EmbeddingRequest):
    try:
        logging.info("Received indexing request")
        if not request.texts:
            raise HTTPException(status_code=400, detail="The 'texts' list cannot be empty.")

        # Call models service to generate embeddings
        model_response = requests.post(f"{MODELS_SERVICE_URL}/embed", json={"texts": request.texts})
        model_response.raise_for_status()
        embeddings = model_response.json().get("embeddings", [])

        if not embeddings:
            raise HTTPException(status_code=500, detail="Failed to generate embeddings")

        # Prepare data for Qdrant
        collection_name = "store_raw_data"
        # headers = {"Authorization": f"Bearer {QDRANT_API_KEY}"}
        ensure_collection_exists(collection_name)
    
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"text": text}
            ) for i, (vector, text) in enumerate(zip(embeddings, request.texts))
        ]

        qdrant_response = qdrant_client.upsert(collection_name=collection_name, points=points)

        if qdrant_response.status != models.UpdateStatus.COMPLETED:
            logging.error(f"Failed to insert data into Qdrant: {qdrant_response}")
            raise HTTPException(status_code=502, detail="Failed to insert data into Qdrant")

        # Log request and response
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO logs (endpoint, request_body, response_body) VALUES (?, ?, ?)",
            ("/index", str(request.dict()), str(qdrant_response))
        )
        conn.commit()
        conn.close()

        logging.info("Successfully indexed data")
        return EmbeddingResponse(message="Data indexed successfully")
    except requests.RequestException as e:
        logging.error(f"Error communicating with models service: {str(e)}")
        raise HTTPException(status_code=502, detail="Bad Gateway: Models service error")
    except Exception as e:
        logging.error(f"Error indexing data: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/index_files", response_model=EmbeddingResponse)
async def index_uploaded_files():
    try:
        logging.info("Received request to index PDF files")

        pdf_files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".pdf")]
        if not pdf_files:
            raise HTTPException(status_code=400, detail="No PDF files found for indexing")

        new_files = [f for f in pdf_files if f not in indexed_files_cache]
        if not new_files:
            logging.info("All PDF files have already been indexed.")
            return EmbeddingResponse(message="No new files to index")

        texts = []
        for pdf_file in new_files:
            file_path = os.path.join(UPLOAD_DIR, pdf_file)
            reader = PdfReader(file_path)
            extracted_text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
            if extracted_text.strip():
                texts.append(extracted_text)
            else:
                logging.warning(f"No text extracted from file: {pdf_file}")

        if not texts:
            raise HTTPException(status_code=400, detail="No valid text extracted from the PDF files.")

        # Call models service to generate embeddings
        model_response = requests.post(f"{MODELS_SERVICE_URL}/embed", json={"texts": texts})
        model_response.raise_for_status()
        embeddings = model_response.json().get("embeddings", [])

        if not embeddings:
            raise HTTPException(status_code=500, detail="Failed to generate embeddings")

        collection_name = "store_raw_data"
        # headers = {"Authorization": f"Bearer {QDRANT_API_KEY}"}

        # Create collection if it does not exist
        ensure_collection_exists(collection_name)

        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload={"file_name": new_files[i], "text": text}
            ) for i, (vector, text) in enumerate(zip(embeddings, texts))
        ]

        qdrant_response = qdrant_client.upsert(collection_name=collection_name, points=points)

        if qdrant_response.status != models.UpdateStatus.COMPLETED:
            logging.error(f"Failed to insert data into Qdrant: {qdrant_response}")
            raise HTTPException(status_code=502, detail="Failed to insert data into Qdrant")

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO logs (endpoint, request_body, response_body) VALUES (?, ?, ?)",
            ("/index_files", str(new_files), str(qdrant_response))
        )
        conn.commit()
        conn.close()

        indexed_files_cache.update(new_files)
        logging.info("Successfully indexed new PDF files")
        return EmbeddingResponse(message="New PDF files indexed successfully")

    except requests.RequestException as e:
        logging.error(f"Error communicating with models service: {str(e)}")
        raise HTTPException(status_code=502, detail="Bad Gateway: Models service error")
    except Exception as e:
        logging.error(f"Error indexing PDF files: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.get("/indexed_files", response_model=List[str])
async def get_indexed_files():
    try:
        logging.info("Received request to retrieve indexed file names")
        
        # Check if the directory exists
        if not os.path.exists(INDEXED_FILES_PATH):
            logging.error(f"Indexed files path does not exist: {INDEXED_FILES_PATH}")
            raise HTTPException(status_code=500, detail="Indexed files directory not found")
        
        # Retrieve sorted list of files in the directory
        indexed_files_list = sorted(os.listdir(INDEXED_FILES_PATH))
        
        # Handle empty directory case
        if not indexed_files_list:
            logging.info("No files have been indexed yet.")
            raise HTTPException(status_code=404, detail="No indexed files found")
        
        logging.info(f"Retrieved {len(indexed_files_list)} indexed files")
        return indexed_files_list
    
    except OSError as e:
        logging.error(f"OS error while accessing indexed files: {str(e)}")
        raise HTTPException(status_code=500, detail="Error accessing indexed files")
    
    except Exception as e:
        logging.error(f"Unexpected error retrieving indexed files: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

@app.post("/index_json", response_model=EmbeddingResponse)
async def index_candidates_from_json(file: UploadFile = File(...)):
    try:
        logging.info("Received request to index candidates from JSON file")

        # Save the uploaded file locally
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Load and parse the saved JSON file
        with open(file_path, "r") as json_file:
            candidates = json.load(json_file)

        if not isinstance(candidates, list):
            raise HTTPException(status_code=400, detail="Invalid JSON format. Expected a list of candidates.")

        texts = [json.dumps(candidate) for candidate in candidates]

        # Call models service to generate embeddings
        model_response = requests.post(f"{MODELS_SERVICE_URL}/embed", json={"texts": texts})
        model_response.raise_for_status()
        embeddings = model_response.json().get("embeddings", [])

        if not embeddings:
            raise HTTPException(status_code=500, detail="Failed to generate embeddings")

        collection_name = "candidates_collection"
        ensure_collection_exists(collection_name)

        # Prepare data for Qdrant
        points = [
            models.PointStruct(
                id=str(uuid.uuid4()),
                vector=vector,
                payload=candidate
            ) for candidate, vector in zip(candidates, embeddings)
        ]

        qdrant_response = qdrant_client.upsert(collection_name=collection_name, points=points)

        if qdrant_response.status != models.UpdateStatus.COMPLETED:
            logging.error(f"Failed to insert data into Qdrant: {qdrant_response}")
            raise HTTPException(status_code=502, detail="Failed to insert data into Qdrant")

        # Log request and response
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO logs (endpoint, request_body, response_body) VALUES (?, ?, ?)",
            ("/index_json", file.filename, str(qdrant_response))
        )
        conn.commit()
        conn.close()

        logging.info("Successfully indexed candidates from JSON file")
        return EmbeddingResponse(message="Candidates indexed successfully")

    except json.JSONDecodeError as e:
        logging.error(f"Invalid JSON file: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid JSON file")
    except requests.RequestException as e:
        logging.error(f"Error communicating with models service: {str(e)}")
        raise HTTPException(status_code=502, detail="Bad Gateway: Models service error")
    except Exception as e:
        logging.error(f"Error indexing candidates: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
