import streamlit as st
import pandas as pd
import sqlite3

# Connect to SQLite database
connection = sqlite3.connect('./db/indexing_service_logs.db')
df = pd.read_sql_query("SELECT * FROM logs", connection)
connection.close()

st.title('Candidate Scoring Dashboard')
st.dataframe(df)