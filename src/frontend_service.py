### Updated `frontend_service.py`
import streamlit as st
import requests
import os

# Fetching service URLs from environment variables with defaults
UPLOAD_SERVICE_URL = os.getenv("UPLOAD_SERVICE_URL", "http://localhost:8002/upload")
INDEX_SERVICE_URL = os.getenv("INDEX_SERVICE_URL", "http://localhost:8000")
SEARCH_SERVICE_URL = os.getenv("SEARCH_SERVICE_URL", "http://localhost:8004/search_candidates")
LLM_SERVICE_URL = os.getenv("LLM_SERVICE_URL", "http://localhost:8005/generate")

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

# Section 2: File Upload for Excel
st.header("Upload Excel Files")
def upload_excel_file():
    excel_file = st.file_uploader("Upload Excel files", type=["xlsx", "xls"])
    if excel_file is not None:
        files = {"file": excel_file}
        response = requests.post(f"{INDEX_SERVICE_URL}/index_excel", files=files)
        if response.status_code == 200:
            st.success(f"Excel file '{excel_file.name}' indexed successfully.")
        else:
            st.error("Failed to index the Excel file.")

upload_excel_file()

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

# Section 5: Chat Interface
st.header("Chat with the Screening Assistant")
def chat_with_llm():
    user_query = st.text_input("Ask a question about the candidates or resumes:")
    if st.button("Send Query") and user_query:
        # Call the LLM service for general queries
        llm_response = requests.post(LLM_SERVICE_URL, json={"prompt": user_query})
        if llm_response.status_code == 200:
            response_text = llm_response.json().get("response", "No response")
            st.write("Assistant:", response_text)
        else:
            st.error("Failed to get a response from the assistant.")

        # Check if the query is about resumes and trigger the search service
        if "resume" in user_query.lower() or "candidate" in user_query.lower():
            search_response = requests.get(SEARCH_SERVICE_URL, params={"query": user_query})
            if search_response.status_code == 200:
                candidates = search_response.json().get("candidates", [])
                st.write("Top candidates:")
                for candidate in candidates:
                    st.write(f"ID: {candidate['id']}, Score: {candidate['score']}")
            else:
                st.error("Failed to retrieve candidates.")

chat_with_llm()
