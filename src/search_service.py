### search_service.py
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
import requests
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
MODELS_SERVICE_URL = os.getenv("MODELS_SERVICE_URL", "http://localhost:8005/embed")

# Initialize FastAPI app
app = FastAPI(title="Search Service", description="Service to search and retrieve top candidates")

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Pydantic models for request and response
class SearchResponse(BaseModel):
    candidates: List[Dict]

@app.get("/search_candidates", response_model=SearchResponse)
async def search_candidates(query: str = Query(..., description="Search query to find top candidates"), top_n: int = Query(30, description="Number of top candidates to retrieve")):
    try:
        logging.info("Received search request")
        
        # Step 1: Call the embedding service to generate an embedding for the query
        embedding_payload = {"texts": [query]}
        logging.debug(f"Embedding payload: {embedding_payload}")
        embedding_response = requests.post(MODELS_SERVICE_URL, json=embedding_payload)
        embedding_response.raise_for_status()
        embedding = embedding_response.json().get("embeddings")[0]
        logging.debug(f"Generated embedding: {embedding}")

        # Step 2: Search in Qdrant using the embedding
        search_payload = {
            "vector": embedding,
            "top": top_n,
            "with_payload": True,
        }
        logging.debug(f"Search payload: {search_payload}")

        collection_name = "candidates_collection"  # Replace with the actual collection name
        headers = {"Authorization": f"Bearer {QDRANT_API_KEY}"}
        response = requests.post(f"{QDRANT_URL}/collections/{collection_name}/points/search", json=search_payload, headers=headers)
        response.raise_for_status()
        results = response.json().get("result", [])

        # Step 3: Prepare candidates for response
        candidates = [
            {
                "id": res["id"],
                "score": res["score"],
                "name": res.get("payload", {}).get("Name", "N/A"),
                "job_title": res.get("payload", {}).get("Job title", "N/A"),
                "location": res.get("payload", {}).get("Job location", "N/A"),
                "summary": res.get("payload", {}).get("Summary", "N/A"),
                "keywords": res.get("payload", {}).get("Keywords", "N/A"),
                "experience": res.get("payload", {}).get("Experiences", "N/A")
            }
            for res in results
        ]
        
        logging.info("Successfully retrieved search results")
        logging.debug(f"Search results: {candidates}")
        return SearchResponse(candidates=candidates)
    except requests.RequestException as e:
        logging.error(f"Error communicating with external services: {str(e)}")
        raise HTTPException(status_code=502, detail="Bad Gateway: External service error")
    except Exception as e:
        logging.error(f"Error during search: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)
