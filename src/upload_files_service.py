from fastapi import FastAPI, UploadFile, File, HTTPException
import os
import shutil
import logging
import uvicorn
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(title="Upload Files Service", description="Service to upload and store files locally")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Directory to store uploaded files
UPLOAD_DIR = os.path.join("src", "main", "docs")
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    try:
        file_path = os.path.join(UPLOAD_DIR, file.filename)
        logging.info(f"Uploading file to {file_path}")
        
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        logging.info(f"Successfully uploaded file: {file.filename}")
        return {"message": f"File '{file.filename}' uploaded successfully"}
    except Exception as e:
        logging.error(f"Error uploading file: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8002)
