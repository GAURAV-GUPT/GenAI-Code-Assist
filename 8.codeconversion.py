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

# Load .env for OpenAI key
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini")

st.set_page_config(page_title="GenAI Full Dev Assistant")
# st.title("🧠 GenAI - BAU Support Assistant")




# --- Mocking the LLM functionality ---
# In a real application, you would replace this with a call to a
# language model API, such as the Gemini API.
# This function simulates the LLM's response.
def run_conversion_model(prompt):
    """
    Simulates a call to a large language model to translate code.
    In a real app, this would use an actual LLM client.
    """
    st.info("🤖 AI is translating your code... This may take a moment.")
    time.sleep(3)  # Simulate API call delay
    
    # Simple, hardcoded mock responses for demonstration.
    if "Python" in prompt:
        return """
# Python translation of the original code
def process_data(input_record):
    # Assuming 'input_record' is a dictionary
    customer_id = input_record.get('cust_id')
    balance = input_record.get('balance')

    # Add business logic here
    if balance > 1000:
        print(f"Customer {customer_id} has a high balance.")
    return f"Processing complete for customer {customer_id}."
"""
    elif "Java" in prompt:
        return """
// Java translation of the original code
public class DataProcessor {
    public String processData(Map<String, String> inputRecord) {
        String customerId = inputRecord.get("cust_id");
        double balance = Double.parseDouble(inputRecord.get("balance"));

        // Add business logic here
        if (balance > 1000) {
            System.out.println("Customer " + customerId + " has a high balance.");
        }
        return "Processing complete for customer " + customerId + ".";
    }
}
"""
    elif ".NET" in prompt:
        return """
// C# translation of the original code
using System;
using System.Collections.Generic;

public class DataProcessor
{
    public string ProcessData(Dictionary<string, string> inputRecord)
    {
        string customerId = inputRecord["cust_id"];
        double balance = double.Parse(inputRecord["balance"]);

        // Add business logic here
        if (balance > 1000)
        {
            Console.WriteLine("Customer " + customerId + " has a high balance.");
        }
        return "Processing complete for customer " + customerId + ".";
    }
}
"""
    return "Error: Could not determine target language."


def main():
    st.set_page_config(page_title="Code Translator", page_icon="📝")
    st.title("Legacy Code Translator ⚙️")
    st.markdown("Use this tool to translate your legacy code (e.g., Cobol, Fortran) into modern languages like Python, Java, or .NET.")

    # --- UI elements for user input ---
    st.subheader("1. Select Target Language")
    target_language = st.selectbox(
        "Choose the language you want to translate the code to:",
        ("Python", "Java", ".NET (C#)")
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
        if source_code:
            # --- Prompt generation and model call ---
            # This is the prompt that would be sent to the LLM.
            prompt_template = """
You are a highly skilled software engineer tasked with migrating legacy code.
Please convert the following code into {target_lang}.
Analyze the original code's logic and functionality to ensure the translated code is accurate and idiomatic for the new language.

Original Code:
```{source_code}```

Translated Code:
"""
            prompt = prompt_template.format(
                target_lang=target_language,
                source_code=source_code
            )
            
            try:
                # Call the mock function to get the translated code
                translated_code = run_conversion_model(prompt)
                
                # --- Displaying the output ---
                st.subheader(f"✅ Translated {target_language} Code")
                if "Python" in target_language:
                    st.code(translated_code, language="python")
                elif "Java" in target_language:
                    st.code(translated_code, language="java")
                else:
                    st.code(translated_code, language="csharp")
                
            except Exception as e:
                st.error(f"❌ An error occurred during translation: {e}")
        else:
            st.warning("Please enter some source code to translate.")

if __name__ == "__main__":
    main()
