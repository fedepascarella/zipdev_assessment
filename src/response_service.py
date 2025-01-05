### response_service.py
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
    intent: str

class ScoringRequest(BaseModel):
    candidates: List[Candidate]

class ScoringResponse(BaseModel):
    response: str

@app.post("/generate_response", response_model=ScoringResponse)
async def generate_response(request: ScoringRequest):
    """Generate a context-aware response using search results and LLM."""
    try:
        logging.info("Starting response generation process")

        # Step 1: Prepare context from candidates
        context = "\n".join([f"Candidate ID: {c.id}, Intent: {c.intent}" for c in request.candidates])
        
        # Step 2: Send context to LLM for response generation
        prompt = (
            f"Based on the following candidates, generate a response for the given job description:\n"
            f"Candidates:\n{context}"
        )
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