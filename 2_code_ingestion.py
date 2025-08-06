import streamlit as st
import pandas as pd
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os

# Load .env
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Streamlit UI setup
st.set_page_config(page_title="Code Ingestion & Semantic Search")
st.title("📥 Code Ingestion + Semantic Search")

# File uploader
uploaded_file = st.file_uploader("Upload your code metadata CSV", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {len(df)} rows from the CSV file.")
    st.dataframe(df)

    # Convert rows to LangChain Documents
    st.info("Embedding and storing documents in vector DB...")
    docs = [
        Document(
            page_content=f"File: {row['file_name']}\nFunction: {row['function_name']}\nCode: {row['code_snippet']}\nCommit: {row['commit_message']}"
        ) for _, row in df.iterrows()
    ]

    # Create vector DB using OpenAI Embeddings + FAISS
    embeddings = OpenAIEmbeddings()
    vectordb = FAISS.from_documents(docs, embeddings)

    st.success("🧠 Documents embedded and vector store ready!")

    # User Query Input
    query = st.text_input("🔎 Ask a question about your codebase (e.g., What does the login function do?)")

    if query:
        with st.spinner("Searching with semantic similarity..."):
            results = vectordb.similarity_search(query, k=1)
            top_result = results[0].page_content
            st.subheader("📌 Top Match Found:")
            st.code(top_result, language="python")
else:
    st.warning("📂 Please upload a CSV file with code metadata to begin.")