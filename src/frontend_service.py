### Updated `frontend_service.py`
import streamlit as st
import requests
import os
import logging
from traceloop.sdk import Traceloop
from traceloop.sdk.decorators import workflow, task

Traceloop.init(disable_batch=True)

# Fetching service URLs from environment variables with defaults
UPLOAD_SERVICE_URL = os.getenv("UPLOAD_SERVICE_URL", "http://localhost:8002/upload")
INDEX_SERVICE_URL = os.getenv("INDEX_SERVICE_URL", "http://localhost:8000")
SEARCH_SERVICE_URL = os.getenv("SEARCH_SERVICE_URL", "http://localhost:8004/search_candidates")
RESPONSE_SERVICE_URL = os.getenv("RESPONSE_SERVICE_URL", "http://localhost:8006/generate_response")
QDRANT_COLLECTION_NAME = os.getenv("QDRANT_COLLECTION_NAME", "store_embeddings")
TRACELOOP_API_KEY = os.getenv("TRACELOOP_API_KEY")


logging.basicConfig(filename='debug.log', level=logging.DEBUG)

st.title("AI-Powered Candidate Screening System")

# Section 1: File Upload for PDF
st.header("Upload Resumes or Job Descriptions")
def upload_file():
    uploaded_file = st.file_uploader("Upload PDF files", type=["pdf"])
    if uploaded_file is not None:
        files = {"file": uploaded_file}
        response = requests.post(UPLOAD_SERVICE_URL, files=files)
        if response.status_code == 200:
            st.success(f"File '{uploaded_file.name}' uploaded successfully.")
        else:
            st.error("Failed to upload file.")

upload_file()

# Section 2: Upload JSON File for Candidates
st.header("Upload Candidate JSON File")
def upload_json_file():
    json_file = st.file_uploader("Upload JSON files", type=["json"])
    if json_file is not None:
        files = {"file": json_file}
        response = requests.post(f"{INDEX_SERVICE_URL}/index_json", files=files)
        if response.status_code == 200:
            st.success(f"JSON file '{json_file.name}' indexed successfully.")
        else:
            st.error("Failed to index the JSON file.")

upload_json_file()

# Section 3: Display Indexed Files
st.header("Indexed Files")
def display_indexed_files():
    response = requests.get(f"{INDEX_SERVICE_URL}/indexed_files")
    if response.status_code == 200:
        files = response.json()
        if files:
            st.write("Indexed files:")
            st.write(files)
        else:
            st.write("No files indexed yet.")
    else:
        st.error("Failed to retrieve indexed files.")

if st.button("Show Indexed Files"):
    display_indexed_files()

# Section 4: Trigger Indexing Process for PDF
st.header("Trigger PDF Indexing Process")
def trigger_indexing():
    if st.button("Index Uploaded PDF Files"):
        response = requests.post(f"{INDEX_SERVICE_URL}/index_files")
        if response.status_code == 200:
            st.success("Indexing process completed successfully.")
        else:
            st.error("Failed to complete the indexing process.")

trigger_indexing()

# Section: Chat Interface
st.header("Chat with the Screening Assistant")
@workflow("chat_with_llm")
def chat_with_llm():
    user_query = st.text_input("Ask a question about the candidates or resumes:")
    if st.button("Send Query") and user_query:
        try:
            # Step 1: Call the search service
            search_response = requests.get(SEARCH_SERVICE_URL, params={"query": user_query, "collection": QDRANT_COLLECTION_NAME})
            if search_response.status_code == 200:
                results = search_response.json().get("candidates", [])
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
                st.write("Top candidates retrieved:")
                for candidate in candidates:
                    st.write(f"ID: {candidate['id']}, Name: {candidate['name']}, Job Title: {candidate['job_title']}, Score: {candidate['score']}")
            else:
                st.error("Failed to retrieve candidates.")
                return

            # Step 2: Call the response service with the user query and search results
            
            response_payload = {"query": user_query, "candidates": candidates}
            print(f"Response payload: {response_payload}")
            llm_response = requests.post(RESPONSE_SERVICE_URL, json=response_payload)
            if llm_response.status_code == 200:
                response_text = llm_response.json().get("response", "No response generated.")
                st.write("Assistant:", response_text)
            else:
                st.error("Failed to get a response from the assistant.")

        except requests.RequestException as e:
            st.error(f"An error occurred: {e}")

chat_with_llm()
