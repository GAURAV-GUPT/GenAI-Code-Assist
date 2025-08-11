import streamlit as st
from langchain.chat_models import ChatOpenAI
from dotenv import load_dotenv
import os

# Load API key from .env
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Initialize model
llm = ChatOpenAI(model="gpt-4o-mini")

# Streamlit app UI
st.set_page_config(page_title="✅ Code Validator")
st.title("✅ GenAI Code Validator")

# User Input
st.subheader("Step 1: Paste the Generated Code")
code_input = st.text_area("Paste Python code you'd like reviewed:", height=300)

if code_input and st.button("🔍 Validate Code"):
    with st.spinner("Analyzing your code..."):
        prompt = f"""You are a senior software engineer.

Please review the following Python code for:
- Syntax errors
- Security flaws (e.g., hardcoded secrets, input validation issues)
- Logical errors or anti-patterns
- Suggestions for refactoring or improvements

Code:
{code_input}
"""
        response = llm.invoke(prompt)
        st.success("✅ Validation Result:")
        st.write(response)
