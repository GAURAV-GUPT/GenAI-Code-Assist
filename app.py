import os
import time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from langchain.chat_models import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage

# Load environment variables
load_dotenv()

# Validate API Key & Project
api_key = os.getenv("OPENAI_API_KEY")
project_id = os.getenv("OPENAI_PROJECT")

if not api_key:
    raise ValueError("❌ OPENAI_API_KEY not found in .env file.")
os.environ["OPENAI_API_KEY"] = api_key

if api_key.startswith("sk-proj-") and not project_id:
    raise ValueError("❌ OPENAI_PROJECT is required for project-based keys.")
if project_id:
    os.environ["OPENAI_PROJECT"] = project_id

# Streamlit UI
st.set_page_config(page_title="GenAI Grid Assistant", layout="wide")
st.title("⚡ GenAI Grid Assistant")
st.markdown("Upload your **grid log CSV** and get an intelligent suggestion for a specific row.")

# Optional CSS
if os.path.exists("style.css"):
    with open("style.css") as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Upload CSV
uploaded_file = st.file_uploader("📄 Upload CSV File", type="csv")

# Load LLM
llm = ChatOpenAI(model="gpt-4", temperature=0.3)

# Process if file is uploaded
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### 📊 Uploaded Data", df)

    # Row selection
    row_index = st.number_input("🔢 Select row number to analyze", min_value=0, max_value=len(df)-1, step=1)

    if st.button("🧠 Get Suggestion for Selected Row"):
        row = df.iloc[row_index]
        log_entry = f"{row.to_dict()}"

        messages = [
            SystemMessage(content="You are an expert Grid Operations Assistant. Interpret grid logs and suggest intelligent next steps."),
            HumanMessage(content=f"Analyze this event log entry:\n{log_entry}")
        ]

        with st.spinner("⏳ Getting suggestion..."):
            try:
                start = time.time()
                response = llm(messages)
                st.success(f"✅ Suggestion ready in {time.time() - start:.2f}s")
                st.markdown(f"**Row {row_index} Data:**\n```\n{log_entry}\n```")
                st.markdown(f"**🔍 Suggestion:** {response.content}")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
