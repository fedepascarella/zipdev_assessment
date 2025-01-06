# zipdev_assesment
ZipDev Assesment

# AI-Powered Candidate Screening System

This project is an AI-powered candidate screening system that processes and indexes candidate resumes, provides context-aware responses, and enables intelligent search and retrieval of candidates. The system consists of multiple microservices working in tandem, a frontend interface, and SQLite for log storage. Additionally, Traceloop is used for tracing LLM workflows, and Ollama is required for running local LLM models.

---

## Services Overview

### 1. **Upload Files Service** (`upload_files_service.py`)
   - **Purpose**: Handles the uploading of files (PDFs, JSON) to the local storage.
   - **Endpoint**: `/upload`
   - **Port**: `8002`

### 2. **Indexing Service** (`indexing_service.py`)
   - **Purpose**: Processes uploaded files, generates embeddings using the models service, and stores them in a Qdrant vector database.
   - **Endpoints**:
     - `/index`: Indexes provided texts.
     - `/index_files`: Indexes uploaded PDF files.
     - `/index_json`: Indexes candidates from a JSON file.
     - `/indexed_files`: Retrieves a list of indexed files.
   - **Port**: `8000`
   - **Database**: Uses SQLite (`indexing_service_logs.db`) to store logs.

### 3. **Search Service** (`search_service.py`)
   - **Purpose**: Searches and retrieves top candidates based on a query using Qdrant.
   - **Endpoint**: `/search_candidates`
   - **Port**: `8004`

### 4. **AI Models Service** (`ai_models_service.py`)
   - **Purpose**: Generates embeddings and LLM responses using locally running Ollama models.
   - **Endpoints**:
     - `/embed`: Generates embeddings for provided texts.
     - `/generate`: Generates responses using the specified LLM model.
   - **Port**: `8005`

### 5. **Response Service** (`response_service.py`)
   - **Purpose**: Generates context-aware responses by combining user queries with candidate data and calling the LLM service.
   - **Endpoint**: `/generate_response`
   - **Port**: `8006`

### 6. **Frontend Service** (`frontend_service.py`)
   - **Purpose**: Provides a Streamlit-based user interface for uploading files, indexing data, and interacting with the AI-powered screening system.
   - **Features**:
     - Upload and index PDF and JSON files.
     - Trigger indexing processes.
     - Display indexed files.
     - Chat with the AI assistant.
   - **LLM Tracing**: Uses Traceloop SDK for tracing workflows and tasks.

### 7. **Excel Cleanup Service** (`excel_clean_up_service.py`)
   - **Purpose**: Cleans and flattens uploaded Excel files, preparing them for indexing.
   - **Interface**: Provides a Streamlit interface for file upload and data preview.

---

## Installation Guide

### Prerequisites

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/fedepascarella/zipdev_assessment/tree/dev
   cd zipdev_assessment
   ```

2. **Python 3.11**
3. **Virtual Environment**: Create and activate a virtual environment.
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
4. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
5. **SQLite**: Ensure SQLite is installed and accessible.
6. **Qdrant**: Set up and configure a Qdrant instance (local or cloud).
7. **Traceloop**: Ensure Traceloop is installed and properly configured.
8. **Ollama**: Install Ollama and ensure it is running locally with the required models.
   ```bash
   brew install ollama
   ollama serve
   ```
   Ensure the following models are available:
   - `nomic-embed-text`
   - `llama3.2`

### Environment Variables

Create a `.env` file in the project root and set the following variables:

```env
UPLOAD_SERVICE_URL=http://localhost:8002
INDEX_SERVICE_URL=http://localhost:8000
SEARCH_SERVICE_URL=http://localhost:8004
MODELS_SERVICE_URL=http://localhost:8005
RESPONSE_SERVICE_URL=http://localhost:8006
QDRANT_URL=<your_qdrant_url>
QDRANT_API_KEY=<your_qdrant_api_key>
TRACELOOP_API_KEY=<your_traceloop_api_key>
EMBEDDINGS_MODEL=nomic-embed-text:latest
LLM_MODEL=llama3.2:3b
```

---

## How the App Works

1. **File Upload**: Users upload files (PDF, JSON) via the frontend.
2. **Indexing**: Uploaded files are processed by the indexing service, which calls the models service to generate embeddings. These embeddings are stored in Qdrant.
3. **Search**: Users can query the system via the chat interface. The search service retrieves the most relevant candidates based on the query.
4. **Response Generation**: The response service combines the search results with the user query and calls the models service to generate a detailed response.
5. **Logs**: All service requests and responses are logged in an SQLite database.
6. **Tracing**: Traceloop traces all LLM-related workflows to provide better observability and debugging.

---

## Usage Instructions

1. **Start All Services**
   Use the `app.py` script to start all services concurrently.
   ```bash
   python app.py
   ```
2. **Access the Frontend**
   Once the services are running, open the Streamlit interface by navigating to:
   ```
   http://localhost:8501
   ```
3. **Upload Files**
   - Upload PDF resumes or JSON candidate data.
   - Trigger indexing processes.
4. **Search and Interact**
   - Use the chat interface to ask questions about candidates.
   - View indexed files and detailed responses.

---

## Contact
For any issues or support, please contact Federico.

