# This code creates a Streamlit application for translating code
# from languages like Cobol or Fortran into modern languages.
# Full GenAI Developer Assistant (Steps 1 to 8 Integrated - Final Fixed)
import os
os.environ['GIT_PYTHON_GIT_EXECUTABLE'] = r"C:\Program Files\Git\cmd\git.exe"  # adjust path as needed
import git
import streamlit as st
import time
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from dotenv import load_dotenv
import pandas as pd
from git import Repo, GitCommandError

# Load environment variables for the API key

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini")

#load_dotenv()
#API_KEY = os.getenv("OPENAI_API_KEY")

# Configure the OpenAI API client
#if API_KEY:
 #   client = OpenAI(api_key=API_KEY)
#else:
 #   st.error("Please set your OPENAI_API_KEY in a .env file.")
client=None
st.set_page_config(page_title="Code Translator", page_icon="�")
st.title("Legacy Code Translator ⚙️")
st.markdown("Use this tool to translate your legacy code (e.g., Cobol, Fortran) into modern languages.")

# --- Function to call the OpenAI API for translation ---
def translate_code_with_openai(source_code, target_language):
    """
    Calls the OpenAI API to translate source code into a target language.
    """
    try:
        # Construct the prompt for the OpenAI model
        prompt = f"""
You are a highly skilled software engineer tasked with migrating legacy code.
Please convert the following code into {target_language}.
Analyze the original code's logic and functionality to ensure the translated code is accurate and idiomatic for the new language.
The original code is in a language like Cobol or Fortran.

Original Code:
```{source_code}```

Translated Code:
"""
        response = client.chat.completions.create(
            model="gpt-4o-mini", # You can choose a different model if you prefer
            messages=[
                {"role": "system", "content": "You are a helpful assistant that translates code."},
                {"role": "user", "content": prompt}
            ]
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ An error occurred during translation: {e}"

# --- UI elements for user input ---
st.subheader("1. Select Target Language")
target_language = st.selectbox(
    "Choose the language you want to translate the code to:",
    ("Python", "Java", "C# (.NET)")
)

st.subheader("2. Enter Source Code")
source_code = st.text_area(
    "Paste your source code here (e.g., Cobol, Fortran)",
    height=300,
    value="""
IDENTIFICATION DIVISION.
PROGRAM-ID. HELLO-WORLD.
DATA DIVISION.
WORKING-STORAGE SECTION.
01 GREETING PIC X(20) VALUE "Hello, World!".
PROCEDURE DIVISION.
    DISPLAY GREETING.
    STOP RUN.
"""
)

st.subheader("3. Generate Translated Code")

if st.button("🚀 Translate Code"):
    if not os.environ["OPENAI_API_KEY"]:
        st.warning("Please set your OPENAI_API_KEY in a .env file to enable translation.")
    elif source_code:
        st.info(f"🤖 AI is translating your code to {target_language}...")
        
        # Call the new function to get the translated code from OpenAI
        translated_code = translate_code_with_openai(source_code, target_language)

        st.subheader(f"✅ Translated {target_language} Code")
        
        # Display the code with the correct syntax highlighting
        if "Python" in target_language:
            st.code(translated_code, language="python")
        elif "Java" in target_language:
            st.code(translated_code, language="java")
        else: # C#
            st.code(translated_code, language="csharp")
    else:
        st.warning("Please enter some source code to translate.")





























