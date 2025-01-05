from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import requests  # Use requests for HTTP calls
import logging
import uvicorn
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
MODELS_SERVICE_URL = os.getenv("MODELS_SERVICE_URL", "http://localhost:8005")

# Initialize FastAPI app
app = FastAPI(title="Intent Extraction Service", description="Service for extracting intents and scoring candidates using Models Service")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Fixed system prompt
SYSTEM_PROMPT = (
    "You are a Human Resources job description intent and scoring assistant. "
    "You will extract data in order to be passed to an embedding model. "
    "You will effectively screen and score candidates for a specific role based on their resumes and other relevant information. "
    "You will apply scoring criteria to rank candidates."
)

# Pydantic model for request and response
class IntentExtractionRequest(BaseModel):
    resumes: List[str]
    job_description: str

class IntentExtractionResponse(BaseModel):
    extracted_intents: List[str]

@app.post("/extract_intents", response_model=IntentExtractionResponse)
def extract_intents(request: IntentExtractionRequest):
    try:
        logging.info("Received request for intent extraction")
        logging.info(f"Received {len(request.resumes)} resumes for intent extraction")
        logging.info(f"Request data: resumes={request.resumes}, job_description={request.job_description}")

        # Validate that resumes list is not empty
        if not request.resumes:
            raise HTTPException(status_code=400, detail="The 'resumes' list cannot be empty.")

        # Validate that job_description is provided
        if not request.job_description.strip():
            raise HTTPException(status_code=400, detail="The 'job_description' cannot be empty.")

        # Prepare prompts for each resume
        prompts = [
            f"System: {SYSTEM_PROMPT}\\nUser: Please extract intents from the following resume for the role '{request.job_description}':\\n{resume}"
            for resume in request.resumes
        ]

        responses = []
        for prompt in prompts:
            response = requests.post(f"{MODELS_SERVICE_URL}/generate", json={"prompt": prompt})
            response.raise_for_status()  # Raise exception for HTTP errors
            llm_response = response.json().get("response", "")
            responses.append(llm_response)

        logging.info("Successfully extracted intents")
        return IntentExtractionResponse(extracted_intents=responses)

    except requests.RequestException as e:
        logging.error(f"Error calling /generate endpoint: {str(e)}")
        raise HTTPException(status_code=502, detail="Error communicating with the LLM service")

    except HTTPException as e:
        logging.error(f"Client error: {e.detail}")
        raise

    except Exception as e:
        logging.error(f"Unexpected error: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8003)
