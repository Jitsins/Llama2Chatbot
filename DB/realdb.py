import os
import sqlite3
import random
import string
from IPython.display import Markdown, display
# from openai import OpenAI
from sqlalchemy import (
    create_engine,
    MetaData,
    Table,
    Column,
    String,
    Integer,
    select,
)
import pandas as pd
from llama_index.core import SQLDatabase
from llama_index.core.query_engine import NLSQLTableQueryEngine
from llama_index.llms.ollama import Ollama
from llama_index.llms.openai import OpenAI

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Document, ServiceContext, Settings
import streamlit as st


def execute_query(query):
    # Connect to SQLite database using the provided JDBC path
    conn = sqlite3.connect(
        r'C:\Users\jitsi\AppData\Roaming\DBeaverData\workspace6\.metadata\sample-database-sqlite-1\Chinook.db')
    cursor = conn.cursor()

    # Execute the query
    cursor.execute(query)

    # Fetch and store the results
    results = cursor.fetchall()
    columns = [description[0] for description in cursor.description]

    # Close the connection
    conn.close()

    # Return the results
    return results, columns


engine = create_engine(
    r'dbpath')
metadata_obj = MetaData()

sql_database = SQLDatabase(engine, include_tables=["Track", "Genre", "Album"])

# llm = Ollama(model="llama3", request_timeout=60.0, system_prompt="You are bot that only responds based on facts and crisply summarizes results. ")
os.environ["OPENAI_API_KEY"] = ""
llm = OpenAI(temperature=0.7, model_name="gpt-4")

# RAG
embed_model_name = "sentence-transformers/all-MiniLM-L6-v2"
embed_model = HuggingFaceEmbedding(model_name=embed_model_name)

service_context = ServiceContext.from_defaults(
    chunk_size=1024, llm=llm, embed_model=embed_model)

st.title("Chat with Your Database")
query_input = st.text_input("Natural language query to chat with your db")

if st.button("Submit"):
    with st.spinner('Processing your query...'):
        query_engine = NLSQLTableQueryEngine(
            sql_database=sql_database, tables=["Track", "Genre", "Album"], service_context=service_context
        )
        response = query_engine.query(query_input)
        st.write(
            f"<b>SQL Query:</b> <br>{response.metadata['sql_query']}", unsafe_allow_html=True)
        st.write(
            f"<b>Response:</b> <br>{response.response}", unsafe_allow_html=True)

        query_resp = execute_query(response.metadata['sql_query'])
        print(query_resp)
        st.write(pd.DataFrame(query_resp[0], columns=query_resp[1]))