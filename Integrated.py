# Full GenAI Developer Assistant (Steps 1 to 11 Integrated - Predictive Maintenance Added)
import os
import streamlit as st
import pandas as pd
from dotenv import load_dotenv
from git import Repo, GitCommandError
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from openai import OpenAI
import numpy as np

# Load .env for OpenAI key
load_dotenv()
# Set OpenAI API key from environment variables
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# --- Initial App Setup ---
st.set_page_config(page_title="AI for IT - Assistant", layout="wide")
st.title("🧠 **AI for IT - Assistant**")
st.markdown("A multi-agent AI assistant for various IT tasks, from ticket analysis to predictive maintenance.")

# Initialize the OpenAI LLM and Client
# It's good practice to do this once at the top
try:
    llm = ChatOpenAI(model="gpt-4o-mini")
    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    openai_initialized = True
except Exception as e:
    st.error(f"❌ Error initializing OpenAI: {e}. Please check your API key.")
    openai_initialized = False
    llm = None
    openai_client = None

# --- Main functions (moved to top-level) ---
def translate_code_with_openai(client, source_code, target_language):
    """
    Calls the OpenAI API to translate source code into a target language.
    """
    if not client:
        return "❌ An error occurred: OpenAI client is not initialized."
    try:
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
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that translates code.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"❌ An error occurred during translation: {e}"

# --- Streamlit UI and Logic ---

step = st.sidebar.radio(
    "**Available Agents:**",
    [
        "1. Ticket Summarization",
        # "2. Codebase Ingestion",
        # "3. Code Search",
        # "4. Modification Plan",
        # "5. Few-Shot Prompt (Optional)",
        "6. Code Generation",
        # "7. Code Validation",
        # "8. Git Commit + Push",
        "9. App Log Analyser",
        "10. Legacy Code Convertor",
        "11. Predictive Maintenance" # New Agent Added
    ],
)

# Initialize session state variables
if "vectordb" not in st.session_state:
    st.session_state.vectordb = None
if "few_shot" not in st.session_state:
    st.session_state.few_shot = ""
if "ticket" not in st.session_state:
    st.session_state.ticket = ""
if "generated_code" not in st.session_state:
    st.session_state.generated_code = ""

if step == "1. Ticket Summarization":
    st.subheader("📝 Summarize a User Story or Ticket")
    ticket = st.text_area(
        "Enter your ticket or user story", value=st.session_state.ticket
    )
    if ticket and llm:
        prompt = PromptTemplate.from_template(
            """Summarize this user story and extract intent, task type, and keywords:

{ticket}"""
        )
        try:
            chain = LLMChain(llm=llm, prompt=prompt)
            summary = chain.run(ticket=ticket)
            st.write(summary)
            st.session_state.ticket = ticket
        except Exception as e:
            st.error(f"❌ Error generating summary: {e}")

elif step == "2. Codebase Ingestion":
    st.subheader("📁 Upload and Embed Codebase")
    file = st.file_uploader("Upload CSV with codebase", type="csv")
    if file and llm:
        try:
            df = pd.read_csv(file)
            st.dataframe(df)
            docs = [
                Document(
                    page_content=f"""File: {r['file_name']}
Function: {r['function_name']}
Code: {r['code_snippet']}
Commit: {r['commit_message']}"""
                )
                for _, r in df.iterrows()
            ]
            embeddings = OpenAIEmbeddings()
            st.session_state.vectordb = FAISS.from_documents(docs, embeddings)
            st.success("✅ Codebase embedded and ready for semantic search!")
        except Exception as e:
            st.error(f"❌ Failed to process file: {e}")

elif step == "3. Code Search":
    st.subheader("🔍 Semantic Code Search")
    if st.session_state.vectordb:
        q = st.text_input("Enter your search query (e.g., 'login handler')")
        if q:
            try:
                res = st.session_state.vectordb.similarity_search(q, k=1)
                st.code(res[0].page_content)
            except Exception as e:
                st.error(f"❌ Search failed: {e}")
    else:
        st.warning("⚠️ Please complete codebase ingestion first.")

elif step == "4. Modification Plan":
    st.subheader("🛠️ Generate a Modification Plan")
    ticket = st.text_area("Enter your user story", value=st.session_state.ticket)
    if ticket and llm:
        mod_prompt = PromptTemplate.from_template(
            """Given this user story:

{ticket}

Propose a step-by-step code modification plan including file/function suggestions."""
        )
        try:
            mod_chain = LLMChain(llm=llm, prompt=mod_prompt)
            mod_plan = mod_chain.run(ticket=ticket)
            st.write("### 🧠 Suggested Modification Plan:")
            st.write(mod_plan)
            st.session_state.plan = mod_plan
        except Exception as e:
            st.error(f"❌ Error generating modification plan: {e}")

elif step == "5. Few-Shot Prompt (Optional)":
    st.subheader("🧠 Fetch Example Functions for Few-Shot Prompting")
    if st.session_state.vectordb:
        example_key = st.text_input("Keyword to search examples")
        if example_key:
            examples = st.session_state.vectordb.similarity_search(example_key, k=2)
            shot_text = "\n".join([doc.page_content for doc in examples])
            st.text_area("📌 Retrieved Examples:", shot_text, height=300)
            st.session_state.few_shot = shot_text
    else:
        st.warning("⚠️ Upload codebase first.")

elif step == "6. Code Generation":
    st.subheader("💻 Generate Updated Code")
    context = st.text_area(
        "Current code or function context",
        value="def login(user): return user.check_password()",
    )
    change = st.text_input(
        "Describe the required change",
        value="Add OTP verification before checking password",
    )
    if context and change and llm:
        prompt = PromptTemplate.from_template(
            """{shots}

Here is the existing code context:
{context}

You need to: {change}

Please generate the updated Python code."""
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        try:
            result = chain.run(
                shots=st.session_state.few_shot, context=context, change=change
            )
            st.code(result, language="python")
            st.session_state.generated_code = result
        except Exception as e:
            st.error(f"❌ Error generating code: {e}")

elif step == "7. Code Validation":
    st.subheader("✅ Validate the Generated Code")
    code = st.text_area(
        "Paste the code to validate", value=st.session_state.generated_code, height=300
    )
    if code and llm:
        review_prompt = f"You are a senior developer. Review this code for syntax, security issues, and suggest improvements:\n\n{code}"
        review = llm.invoke(review_prompt)
        st.write(review)

elif step == "8. Git Commit + Push":
    st.subheader("🚀 Commit and Push Code to Git")
    repo_path = st.text_input("Local Git repo path", value="./")
    file_path = st.text_input("Relative path to code file", value="generated_code.py")
    commit_msg = st.text_input(
        "Commit message", value="feat: apply GenAI-generated update"
    )
    if st.button("Push to Git"):
        try:
            # Validate local repo path
            abs_repo_path = os.path.abspath(repo_path)
            if not os.path.isdir(abs_repo_path):
                st.error(f"❌ The specified repo path does not exist: {abs_repo_path}")
                st.stop()

            if not os.path.isdir(os.path.join(abs_repo_path, ".git")):
                st.error(
                    f"❌ This folder is not a Git repository (missing .git): {abs_repo_path}"
                )
                st.stop()

            # Validate file path
            full_file_path = os.path.join(abs_repo_path, file_path)
            if not os.path.exists(full_file_path):
                # If the file doesn't exist, create it from generated code (optional)
                if st.session_state.generated_code:
                    with open(full_file_path, "w", encoding="utf-8") as f:
                        f.write(st.session_state.generated_code)
                    st.info(f"📝 {file_path} was created from generated code.")
                else:
                    st.error(f"❌ File not found: {full_file_path}")
                    st.stop()

            # Proceed with Git commit + push
            repo = Repo(abs_repo_path)
            repo.git.add(file_path)
            repo.index.commit(commit_msg)
            repo.remote(name="origin").push()
            st.success("✅ Code committed and pushed successfully!")

        except GitCommandError as e:
            st.error(f"❌ Git command error:\n{e}")
        except Exception as e:
            st.error(f"❌ Unexpected error:\n{e}")

elif step == "9. App Log Analyser":
    st.subheader("📝 Summarize logs")
    logs = st.text_area("Enter your logs here (App/Web and DB)...", value=st.session_state.ticket)
    if logs and llm:
        prompt = PromptTemplate.from_template(
            """Analyze the following log data for the application service 'X'. Identify the most likely root cause of the incident that occurred between [Start Time] and [End Time]. Provide a detailed explanation of the causal chain of events, and suggest at least three specific improvements to prevent a recurrence and improve future troubleshooting efforts:
{logs}"""
        )
        try:
            chain = LLMChain(llm=llm, prompt=prompt)
            summary = chain.run(logs=logs)
            st.write(summary)
            st.session_state.ticket = logs
        except Exception as e:
            st.error(f"❌ Error generating summary: {e}")

# --- Legacy Code Conversion Block (Fixed) ---
elif step == "10. Legacy Code Convertor":
    st.title("Legacy Code Translator ⚙️")
    st.markdown("Use this tool to translate your legacy code (e.g., Cobol, Fortran) into modern languages.")
    
    # Check if the client was initialized successfully
    if not openai_initialized:
        st.warning("⚠️ OpenAI client is not initialized. Please set your OPENAI_API_KEY in a `.env` file.")
    else:
        st.subheader("1. Select Target Language")
        target_language = st.selectbox(
            "Choose the language you want to translate the code to:",
            ("Python", "Java", "C# (.NET)"),
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
""",
        )

        st.subheader("3. Generate Translated Code")
        if st.button("🚀 Translate Code"):
            if not source_code:
                st.warning("Please enter some source code to translate.")
            else:
                with st.spinner(f"🤖 AI is translating your code to {target_language}..."):
                    translated_code = translate_code_with_openai(openai_client, source_code, target_language)

                st.subheader(f"✅ Translated {target_language} Code")

                # Display the code with the correct syntax highlighting
                if "Python" in target_language:
                    st.code(translated_code, language="python")
                elif "Java" in target_language:
                    st.code(translated_code, language="java")
                else:
                    st.code(translated_code, language="csharp")

# --- NEW: Predictive Maintenance Agent ---
elif step == "11. Predictive Maintenance":
    st.subheader("🏭 Predictive Maintenance Agent for Plant Floor")
    st.markdown("Upload sensor data from factory machinery to predict potential failures.")

    # 1. Provide a sample data format for the user
    st.info("""
    **Data Format Guide:**
    Please upload a CSV file with the following columns:
    - `timestamp`: The date and time of the reading.
    - `machine_id`: A unique identifier for the machine (e.g., 'CNC-001', 'ROBOT-ARM-05').
    - `vibration_hz`: Vibration level in Hertz.
    - `temperature_c`: Temperature in Celsius.
    - `power_kw`: Power consumption in Kilowatts.
    - `error_code`: Any error code reported by the machine (0 if none).
    """)

    # 2. File uploader
    uploaded_file = st.file_uploader("Upload your sensor data (CSV)", type="csv")

    if uploaded_file is not None and llm:
        try:
            # Read the data into a pandas DataFrame
            df = pd.read_csv(uploaded_file)
            st.write("### Sensor Data Preview:")
            st.dataframe(df.head())

            # Convert dataframe to a string format suitable for the LLM
            data_string = df.to_string(index=False)

            # 3. Create a detailed prompt for the LLM
            maintenance_prompt = PromptTemplate.from_template(
                """
You are an expert Predictive Maintenance AI Agent for an automotive factory floor.
Your task is to analyze the following real-time sensor data from our machinery and provide a detailed maintenance report.

**Sensor Data:**
```
{sensor_data}
```

**Instructions:**
1.  **Analyze the Data:** Carefully examine the trends in vibration, temperature, and power consumption for each machine. Pay close attention to any anomalies, spikes, or gradual increases that deviate from normal operating parameters.
2.  **Identify At-Risk Machinery:** Clearly state which machine (by `machine_id`) is showing the strongest indicators of an impending failure.
3.  **Provide Root Cause Analysis:** Explain *why* you believe this machine is at risk. Reference specific data points and trends from the provided data (e.g., "Vibration for CNC-002 has increased by 15% over the last 24 hours while temperature is also rising, suggesting bearing wear.").
4.  **Estimate Time to Failure (TTF):** Provide a qualitative estimate of the urgency (e.g., "Critical: Failure likely within 24-48 hours", "Warning: Maintenance recommended within the next 7 days", "Stable: No immediate action required").
5.  **Recommend Actionable Steps:** List clear, specific maintenance actions to be taken. For example:
    - "Schedule immediate inspection of the main spindle bearing on CNC-002."
    - "Lubricate the primary joints of ROBOT-ARM-05."
    - "Order a replacement for part number 74B-221."
6.  **Format your response** as a clear, professional report. Use markdown for headings and bullet points.
"""
            )

            # 4. Run the analysis
            if st.button("🤖 Analyze and Predict Failures"):
                with st.spinner("Analyzing sensor data and predicting outcomes..."):
                    chain = LLMChain(llm=llm, prompt=maintenance_prompt)
                    report = chain.run(sensor_data=data_string)
                    st.write("###  Predictive Maintenance Report:")
                    st.markdown(report)

        except Exception as e:
            st.error(f"❌ An error occurred while processing the file: {e}")

