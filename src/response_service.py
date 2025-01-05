# Updating Response Service
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict
import uvicorn
import requests
import logging
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
SEARCH_SERVICE_URL = os.getenv("SEARCH_SERVICE_URL", "http://localhost:8004/search_candidates")
MODELS_SERVICE_URL = os.getenv("MODELS_SERVICE_URL", "http://localhost:8005/generate")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "candidates_collection")

# Initialize FastAPI app
app = FastAPI(
    title="Candidate Response Service",
    description="Service to generate context-aware responses using LLM"
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Pydantic models for request and response
class Candidate(BaseModel):
    id: str
    score: float
    name: str
    job_title: str
    location: str
    summary: str
    keywords: str
    experience: str


class GenerateResponseRequest(BaseModel):
    query: str
    candidates: List[Candidate]

class ScoringResponse(BaseModel):
    response: str

@app.post("/generate_response", response_model=ScoringResponse)
async def generate_response(request: GenerateResponseRequest):
    """Generate a context-aware response using search results and LLM."""
    try:
        logging.info("Starting response generation process")

        # Step 1: Prepare detailed context from search results
        context = "\n".join([
            f"Candidate ID: {c.id}, Name: {c.name}, Job Title: {c.job_title}, "
            f"Location: {c.location}, Summary: {c.summary}, Keywords: {c.keywords}, "
            f"Experience: {c.experience}, Score: {c.score}"
            for c in request.candidates
        ])

        # Step 2: Send context to LLM for response generation
        prompt = f"Based on the following candidates, answer the query:\n{context}\nUser Query: {request.query}"
        model_response = requests.post(MODELS_SERVICE_URL, json={"prompt": prompt})
        model_response.raise_for_status()
        llm_response = model_response.json().get("response", "No response generated.")
        
        logging.info("Successfully generated response using LLM")
        return ScoringResponse(response=llm_response)
    except requests.RequestException as e:
        logging.error(f"Error communicating with external services: {str(e)}")
        raise HTTPException(status_code=502, detail="Bad Gateway: External service error")
    except Exception as e:
        logging.error(f"Error during response generation: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8006)