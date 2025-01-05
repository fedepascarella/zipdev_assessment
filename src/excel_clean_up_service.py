import streamlit as st
import pandas as pd
import json
import numpy as np
from io import BytesIO


def clean_excel_for_qdrant(file_buffer: BytesIO) -> list[dict]:
    """
    Cleans and flattens an uploaded Excel file, ensuring timestamps are JSON-serializable
    and multiindex columns are flattened.
    
    Parameters:
        file_buffer (BytesIO): Uploaded Excel file buffer.
    
    Returns:
        list[dict]: Cleaned data as a list of dictionaries.
    """
    try:
        # Load the Excel file into a DataFrame
        df = pd.read_excel(file_buffer)
    except Exception as e:
        raise ValueError(f"Error reading Excel file: {e}")

    # Flatten MultiIndex columns, if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join(map(str, col)).strip() for col in df.columns]
    else:
        df.columns = [str(col) for col in df.columns]

    # Fill NaN values appropriately
    for col in df.columns:
        if np.issubdtype(df[col].dtype, np.datetime64):  # Convert datetime to ISO format
            df[col] = df[col].fillna("").apply(lambda x: x.isoformat() if pd.notnull(x) else "")
        elif np.issubdtype(df[col].dtype, np.number):  # Handle numeric columns
            df[col] = df[col].fillna(0)
        else:  # Handle other data types by replacing NaN with empty strings
            df[col] = df[col].fillna("")

    # Convert DataFrame to list of dictionaries
    payloads = df.to_dict(orient="records")
    
    return payloads


def main():
    """
    Main function for the Streamlit app.
    """
    st.title("Excel Cleanup Service for Qdrant")
    st.write(
        """
        This tool allows you to upload an Excel file, clean its content, and prepare it for
        indexing into Qdrant or other applications.
        """
    )

    # File uploader
    uploaded_file = st.file_uploader("Upload your Excel file", type=["xlsx"])

    if uploaded_file:
        st.write("Processing your file...")
        cleaned_payloads = clean_excel_for_qdrant(uploaded_file)

        if cleaned_payloads:
            st.success("File processed successfully!")
            st.write(f"Preview of cleaned data (showing first 5 rows):")
            st.json(cleaned_payloads[:5])  # Show a preview of the first 5 rows

            # Option to download the cleaned data as a JSON file
            json_data = json.dumps(cleaned_payloads, ensure_ascii=False, indent=4)
            st.download_button(
                label="Download cleaned data as JSON",
                data=json_data,
                file_name="cleaned_data.json",
                mime="application/json",
            )


if __name__ == "__main__":
    main()
