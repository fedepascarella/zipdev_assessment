from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
import logging
import os
from dotenv import load_dotenv
import ollama  # Import the Ollama library

# Load environment variables
load_dotenv()
EMBEDDINGS_MODEL = os.getenv("EMBEDDINGS_MODEL", "nomic-embed-text:latest")
LLM_MODEL = os.getenv("LLM_MODEL", "llama3.2:3b")

# Initialize FastAPI app
app = FastAPI(title="Models Service", description="Service for generating embeddings and LLM responses using Ollama models")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Pydantic models for request and response
class EmbeddingRequest(BaseModel):
    texts: List[str]

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]

class LLMRequest(BaseModel):
    prompt: str

class LLMResponse(BaseModel):
    response: str

@app.post("/embed", response_model=EmbeddingResponse)
async def generate_embeddings(request: EmbeddingRequest):
    """Generate embeddings using the Ollama embedding model."""
    try:
        logging.info("Received request to generate embeddings using model %s", EMBEDDINGS_MODEL)
        # Use the Ollama embedding model
        response = ollama.embed(model=EMBEDDINGS_MODEL, input=request.texts)
        embeddings = response.get("embeddings", [])
        logging.info("Successfully generated embeddings")
        return EmbeddingResponse(embeddings=embeddings)
    except ollama.OllamaError as e:
        logging.error(f"Error generating embeddings: {str(e)}")
        raise HTTPException(status_code=502, detail="Error generating embeddings")

@app.post("/generate", response_model=LLMResponse)
async def generate_llm_response(request: LLMRequest):
    """Generate a response using the Ollama LLM model."""
    try:
        logging.info("Received request to generate LLM response using model %s", LLM_MODEL)
        # Use the Ollama LLM model
        response = ollama.generate(model=LLM_MODEL, prompt=request.prompt)
        llm_response = response.get("response", "")
        logging.info("Successfully generated LLM response")
        return LLMResponse(response=llm_response)
    except ollama.OllamaError as e:
        logging.error(f"Error generating LLM response: {str(e)}")
        raise HTTPException(status_code=502, detail="Error generating LLM response")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8005)
