from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from typing import List, Dict
import sqlite3
import logging
import uvicorn
import requests
import os
import ollama
import pandas as pd
from PyPDF2 import PdfReader
from qdrant_client import QdrantClient, models
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
INTENT_SERVICE_URL = os.getenv("INTENT_SERVICE_URL", "http://localhost:8003/extract_intents")
MODELS_SERVICE_URL = os.getenv("MODELS_SERVICE_URL", "http://localhost:8005")
UPLOAD_DIR = os.path.join("src", "main", "docs")

# Initialize FastAPI app
app = FastAPI(title="Indexing Service", description="Indexing Service for generating and storing embeddings from extracted intents")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# SQLite Database setup
db_path = "./db/indexing_service_logs.db"

# In-memory cache to store indexed files
indexed_files_cache = set()

# Qdrant client initialization
qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)


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
        if not request.texts:  # Check if request.texts is empty
            raise HTTPException(status_code=400, detail="The 'texts' list cannot be empty.")

        # Call intent extraction service
        intent_payload = {
            "resumes": request.texts,
            "job_description": "Please extract intents from these resumes"
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(f"{INTENT_SERVICE_URL}", json=intent_payload, headers=headers)
        response.raise_for_status()
        intents = response.json().get("extracted_intents", [])
        
        # Generate embeddings using intents
        response = requests.post(f"{MODELS_SERVICE_URL}/embed", json={"texts": intents})
        response.raise_for_status()  # Raise an error if the request fails
        embeddings = response.json().get("embeddings", [])

        
        # Prepare data for Qdrant
                # Prepare Qdrant collection creation payload
        collection_name = "store_embeddings"
        create_collection_payload = {
            "name": collection_name,
            "vectors": {
                "size": 768,  # Assuming all embeddings have the same size
                "distance": "Cosine"  # Metric for similarity search
            }
        }

        # Create the collection in Qdrant (if it doesn't already exist)
        headers = {"Authorization": f"Bearer {QDRANT_API_KEY}"}
        try:
            qdrant_create_response = requests.put(
                f"{QDRANT_URL}/collections/{collection_name}",
                json=create_collection_payload,
                headers=headers
            )
            qdrant_create_response.raise_for_status()
            logging.info(f"Collection '{collection_name}' created or already exists in Qdrant.")
        except requests.RequestException as e:
            logging.error(f"Failed to create collection '{collection_name}' in Qdrant: {e}")
            raise HTTPException(status_code=502, detail="Failed to create Qdrant collection")

        # Prepare data for inserting embeddings into Qdrant
        insert_payload = {
            "points": [
                {"id": str(i), "vector": vector, "payload": {"intent": intents[i]}}
                for i, vector in enumerate(embeddings)
            ]
        }

        # Insert embeddings into the Qdrant collection
        try:
            qdrant_response = requests.post(
                f"{QDRANT_URL}/collections/{collection_name}/points",
                json=insert_payload,
                headers=headers
            )
            qdrant_response.raise_for_status()
            logging.info("Successfully inserted embeddings into Qdrant")
        except requests.RequestException as e:
            logging.error(f"Error inserting embeddings into Qdrant: {e}")
            raise HTTPException(status_code=502, detail="Failed to insert embeddings into Qdrant")

        
        # Log request and response
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO logs (endpoint, request_body, response_body) VALUES (?, ?, ?)",
            ("/index", str(request.dict()), qdrant_response.text)
        )
        conn.commit()
        conn.close()

        logging.info("Successfully indexed embeddings")
        return EmbeddingResponse(message="Embeddings indexed successfully")
    except requests.RequestException as e:
        logging.error(f"Error communicating with external services: {str(e)}")
        raise HTTPException(status_code=502, detail="Bad Gateway: External service error")
    except Exception as e:
        logging.error(f"Error generating and storing embeddings: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@app.post("/index_files", response_model=EmbeddingResponse)
async def index_uploaded_files():
    try:
        logging.info("Received request to index PDF files")

        # Get list of PDF files in the upload directory
        pdf_files = [f for f in os.listdir(UPLOAD_DIR) if f.endswith(".pdf")]
        if not pdf_files:
            raise HTTPException(status_code=400, detail="No PDF files found for indexing")

        # Filter out already indexed files
        new_files = [f for f in pdf_files if f not in indexed_files_cache]
        if not new_files:
            logging.info("All PDF files have already been indexed.")
            return EmbeddingResponse(message="No new files to index")

        # Extract text from new PDF files
        texts = []
        for pdf_file in new_files:
            file_path = os.path.join(UPLOAD_DIR, pdf_file)
            reader = PdfReader(file_path)
            extracted_text = "".join(page.extract_text() for page in reader.pages if page.extract_text())
            if extracted_text.strip():  # Only include non-empty text
                texts.append(extracted_text)
            else:
                logging.warning(f"No text extracted from file: {pdf_file}")

        if not texts:
            raise HTTPException(status_code=400, detail="No valid text extracted from the PDF files.")

        # Correct payload structure for intent extraction
        intent_payload = {
            "resumes": texts,
            "job_description": "Please extract intents from these uploaded PDF files"
        }

        logging.info(f"Sending payload to intent extraction service: {intent_payload}")

        # Call intent extraction service
        response = requests.post(f"{INTENT_SERVICE_URL}", json=intent_payload)
        response.raise_for_status()
        intents = response.json().get("extracted_intents", [])

        # Generate embeddings using extracted intents
        response = requests.post(f"{MODELS_SERVICE_URL}/embed", json={"texts": intents})
        response.raise_for_status()
        embeddings = response.json().get("embeddings", [])

        if not embeddings:
            raise HTTPException(status_code=500, detail="No embeddings generated")

        # Prepare data for Qdrant
        collection_name = "store_embeddings"
        
        # Create the collection in Qdrant (if it doesn't already exist) 
        headers = {"Authorization": f"Bearer {QDRANT_API_KEY}"}
        try:
            # Use the Qdrant 'exists' endpoint to check if the collection already exists
            qdrant_check_response = requests.get(
                f"{QDRANT_URL}/collections/{collection_name}/exists",
                headers=headers
            )
            qdrant_check_response.raise_for_status()
            exists_response = qdrant_check_response.json()

            if not exists_response.get("result", False):
                # Collection does not exist, create it
                create_collection_payload = {
                    "name": collection_name,
                    "vectors": {
                        "size": 768,
                        "distance": "Cosine"
                    }
                }
                qdrant_create_response = requests.put(
                    f"{QDRANT_URL}/collections/{collection_name}",
                    json=create_collection_payload,
                    headers=headers
                )
                qdrant_create_response.raise_for_status()
                logging.info(f"Collection '{collection_name}' successfully created in Qdrant.")
            else:
                logging.info(f"Collection '{collection_name}' already exists in Qdrant.")
        except requests.RequestException as e:
            logging.error(f"Failed to check or create collection '{collection_name}' in Qdrant: {e}")
            raise HTTPException(status_code=502, detail="Failed to check or create Qdrant collection")


        # Insert embeddings into the Qdrant collection
        points = [
            models.PointStruct(
                id=i,
                vector=vector,
                payload={"intent": intents[i]}
            ) for i, vector in enumerate(embeddings)
        ]

        qdrant_response = qdrant_client.upsert(collection_name=collection_name, points=points)
        # Check if the operation was successful
       # Check if the operation was successful by verifying the status
        if qdrant_response.status == models.UpdateStatus.COMPLETED:
            logging.info("Successfully inserted embeddings into Qdrant")
        else:
            logging.error(f"Failed to insert embeddings into Qdrant: {qdrant_response}")
            raise HTTPException(status_code=502, detail="Failed to insert embeddings into Qdrant")


        # # Insert embeddings into the Qdrant collection
        # try:
        #     qdrant_response = requests.put(
        #         f"{QDRANT_URL}/collections/{collection_name}/points",
        #         json=insert_payload,
        #         headers=headers
        #     )
        #     qdrant_response.raise_for_status()
        #     logging.info("Successfully inserted embeddings into Qdrant")
        # except requests.RequestException as e:
        #     logging.error(f"Error inserting embeddings into Qdrant: {e}")
        #     raise HTTPException(status_code=502, detail="Failed to insert embeddings into Qdrant")

        # Log request and response
        # Log request and response
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO logs (endpoint, request_body, response_body) VALUES (?, ?, ?)",
            ("/index_files", str(new_files), str(qdrant_response))  # Use str(qdrant_response) instead of qdrant_response.text
        )
        conn.commit()
        conn.close()

        # Update cache with newly indexed files
        indexed_files_cache.update(new_files)
        logging.info("Successfully indexed new PDF files")
        return EmbeddingResponse(message="New PDF files indexed successfully")

    except requests.RequestException as e:
        logging.error(f"Error communicating with external services: {str(e)}")
        raise HTTPException(status_code=502, detail="Bad Gateway: External service error")
    except Exception as e:
        logging.error(f"Error indexing PDF files: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")



@app.get("/indexed_files", response_model=List[str])
async def get_indexed_files():
    """
    Endpoint to retrieve the names of all indexed files.
    
    Returns:
        List[str]: A list of indexed file names.
    """
    try:
        logging.info("Received request to retrieve indexed file names")
        
        # Convert indexed files cache to a sorted list for consistent output
        indexed_files_list = sorted(indexed_files_cache)
        
        if not indexed_files_list:
            logging.info("No files have been indexed yet.")
            raise HTTPException(status_code=404, detail="No indexed files found")

        logging.info(f"Retrieved {len(indexed_files_list)} indexed files")
        return indexed_files_list
    except Exception as e:
        logging.error(f"Error retrieving indexed files: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


# Add a new endpoint for indexing Excel files
@app.post("/index_excel", response_model=EmbeddingResponse)
async def index_excel_file(request: Request):
    try:
        logging.info("Received request to index Excel file")

        # Ensure request contains a file
        form = await request.form()
        if "file" not in form:
            raise HTTPException(status_code=400, detail="No file provided in the request.")
        
        excel_file = form["file"].file
        file_name = form["file"].filename

        # Read the Excel file into a DataFrame
        df = pd.read_excel(excel_file)
        if df.empty:
            raise HTTPException(status_code=400, detail="The provided Excel file is empty.")
        
        logging.info(f"Successfully read Excel file: {file_name} with {len(df)} rows")

        # Extract rows as payloads
        payloads = df.to_dict(orient="records")
        texts = [str(payload) for payload in payloads]  # Convert each row to string for intent extraction

        # Call intent extraction service
        intent_payload = {
            "resumes": texts,
            "job_description": "Please extract intents from these Excel rows"
        }
        headers = {"Content-Type": "application/json"}
        response = requests.post(f"{INTENT_SERVICE_URL}", json=intent_payload, headers=headers)
        response.raise_for_status()
        intents = response.json().get("extracted_intents", [])

        # Call embedding service
        response = requests.post(f"{MODELS_SERVICE_URL}/embed", json={"texts": intents})
        response.raise_for_status()
        embeddings = response.json().get("embeddings", [])

        if not embeddings:
            raise HTTPException(status_code=500, detail="No embeddings generated")

        # Prepare data for Qdrant
        collection_name = "store_embeddings"
        headers = {"Authorization": f"Bearer {QDRANT_API_KEY}"}

        points = [
            models.PointStruct(
                id=i,
                vector=vector,
                payload={"intent": intents[i], "source": file_name}
            ) for i, vector in enumerate(embeddings)
        ]

        qdrant_response = qdrant_client.upsert(collection_name=collection_name, points=points)

        if qdrant_response.status != models.UpdateStatus.COMPLETED:
            logging.error(f"Failed to insert embeddings into Qdrant: {qdrant_response}")
            raise HTTPException(status_code=502, detail="Failed to insert embeddings into Qdrant")

        # Log the request and response
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO logs (endpoint, request_body, response_body) VALUES (?, ?, ?)",
            ("/index_excel", file_name, str(qdrant_response))
        )
        conn.commit()
        conn.close()

        logging.info(f"Successfully indexed Excel file: {file_name}")
        return EmbeddingResponse(message=f"Excel file '{file_name}' indexed successfully")

    except pd.errors.EmptyDataError:
        logging.error("Empty Excel file provided")
        raise HTTPException(status_code=400, detail="Empty Excel file provided")
    except requests.RequestException as e:
        logging.error(f"Error communicating with external services: {str(e)}")
        raise HTTPException(status_code=502, detail="Bad Gateway: External service error")
    except Exception as e:
        logging.error(f"Error indexing Excel file: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
