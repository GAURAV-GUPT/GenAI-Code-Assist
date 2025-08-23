# Full GenAI Developer Assistant (Steps 1 to 16 Integrated)
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
st.set_page_config(page_title="AI Assistant", layout="wide")
st.title("🧠 **AI - Assistant**")
st.markdown("A multi-agent AI assistant for various tasks, from ticket analysis to remote diagnostics.")

# Initialize the OpenAI LLM and Client
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

# --- Helper functions for SDLC Agents ---
def brd_to_user_stories(brd_content):
    """
    Generates user stories from a Business Requirements Document (BRD) using an LLM.
    """
    if not llm:
        return "LLM is not initialized. Cannot generate user stories."

    prompt = PromptTemplate.from_template(
        """
You are a product manager. Your task is to convert a Business Requirements Document (BRD) into a list of clear, concise user stories.
Each user story should follow the format: "As a [user], I want to [action], so that I can [goal]."

Here is the BRD content:
{brd_content}

Please provide the user stories:
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(brd_content=brd_content)


def user_stories_to_acceptance_criteria(user_stories):
    """
    Generates acceptance criteria from a list of user stories using an LLM.
    """
    if not llm:
        return "LLM is not initialized. Cannot generate acceptance criteria."

    prompt = PromptTemplate.from_template(
        """
You are a QA analyst. Your task is to generate detailed, verifiable acceptance criteria for each of the following user stories.
For each user story, provide a list of criteria that must be met for the story to be considered complete.
Format your response clearly with headings for each user story and a bulleted list for the acceptance criteria.

Here are the user stories:
{user_stories}

Please provide the acceptance criteria:
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(user_stories=user_stories)


def user_stories_to_test_cases(user_stories):
    """
    Generates test cases from a list of user stories using an LLM.
    """
    if not llm:
        return "LLM is not initialized. Cannot generate test cases."

    prompt = PromptTemplate.from_template(
        """
You are a QA engineer. Your task is to generate detailed test cases for each of the following user stories.
For each user story, provide at least one positive test case and one negative/edge case.
Format your response clearly with headings for each user story and bullet points for the test cases.

Here are the user stories:
{user_stories}

Please provide the test cases:
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(user_stories=user_stories)


def generate_code_from_requirements(user_stories, acceptance_criteria, language="Python"):
    """
    Generates source code from user stories and acceptance criteria using an LLM.
    """
    if not llm:
        return "LLM is not initialized. Cannot generate code."

    prompt = PromptTemplate.from_template(
        """
You are an expert software developer. Your task is to write {language} code that fulfills the requirements outlined in the following user stories and their acceptance criteria.
The code should be well-structured, efficient, and include comments where necessary.

Here are the user stories:
{user_stories}

Here is the acceptance criteria:
{acceptance_criteria}

Please provide the final {language} code:
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(
        user_stories=user_stories,
        acceptance_criteria=acceptance_criteria,
        language=language,
    )

# --- Helper functions for Sourcing Agent ---
def run_supplier_research_agent(llm, part_name):
    """Identifies top 3 suppliers for a given automotive part."""
    prompt = PromptTemplate.from_template(
        """
You are a Market Research Agent for a major UK automotive company.
Your task is to identify the top 3 potential suppliers for the following part: **{part_name}**.

For each supplier, provide:
- A realistic company name.
- A brief profile (e.g., focus on quality, cost-effectiveness, innovation, location).
- An estimated per-unit price range in GBP (£).

Format your output clearly in markdown.
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(part_name=part_name)

def run_negotiation_agent(llm, part_name, supplier_profile, target_price):
    """Simulates a negotiation between a sourcing manager and a supplier."""
    prompt = PromptTemplate.from_template(
        """
You are a sophisticated negotiation simulation system. You will orchestrate a dialogue between two AI agents:
- **Agent A (Sourcing Manager):** Represents a large UK automotive firm. Their goal is to secure a per-unit price for **'{part_name}'** at or below their target of **£{target_price}**. They can use leverage points like high-volume orders and long-term partnership potential.
- **Agent B (Sales Director):** Represents the supplier, whose profile is: {supplier_profile}. Their goal is to get the best possible price while securing the contract.

**Instructions:**
1.  Generate a realistic, back-and-forth negotiation transcript.
2.  The dialogue must include an opening offer, at least two rounds of counter-offers, and a final conclusion.
3.  After the transcript, provide a final summary on a new line.

**Output Format:**

**Negotiation Transcript:**
* **Sourcing Manager:** [Opening line]
* **Sales Director:** [Response]
* ...

**Final Summary:**
* **Outcome:** [State whether an agreement was reached and at what final price.]
* **Recommendation:** [Provide a brief recommendation for the Sourcing Manager.]
"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(part_name=part_name, supplier_profile=supplier_profile, target_price=target_price)



# --- NEW: Automotive Campaigns Agent Functions ---
def run_market_research_agent(llm, product_data, competitor_data):
    prompt = PromptTemplate.from_template(
        """You are a market research analyst. Based on the following product and competitor data, write a strategic brief for an automotive campaign. The brief should identify the target audience, the unique selling proposition (USP), and a recommended positioning statement.
        Product Data: {product_data}
        Competitor Data: {competitor_data}"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(product_data=product_data, competitor_data=competitor_data)

def run_creative_agent(llm, strategic_brief):
    prompt = PromptTemplate.from_template(
        """You are a creative director. Based on this strategic brief, develop a creative concept for an automotive campaign. The concept should include a core message, a campaign slogan, and key visual ideas.
        Strategic Brief: {strategic_brief}"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(strategic_brief=strategic_brief)

def run_content_agent(llm, creative_concept):
    prompt = PromptTemplate.from_template(
        """You are a copywriter. Based on this creative concept, generate ad copy for social media (Twitter and Instagram) and a short video script (15 seconds) for the campaign.
        Creative Concept: {creative_concept}"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(creative_concept=creative_concept)

def run_execution_agent(llm, digital_assets):
    prompt = PromptTemplate.from_template(
        """You are a front-end developer. Generate a simple HTML and CSS code for a landing page based on the following digital assets. The page should include a hero section with the slogan, and a simple lead capture form (name, email).
        Digital Assets: {digital_assets}"""
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    return chain.run(digital_assets=digital_assets)


# --- Streamlit UI and Logic ---
step = st.sidebar.radio(
    "**Available Agents:**",
    [
        "1. Ticket Summarization",
        "2. Codebase Ingestion",
        "3. Code Search",
        "4. Modification Plan",
        "5. Few-Shot Prompt (Optional)",
        "6. Code Generation",
        "7. Code Validation",
        "8. Git Commit + Push",
        "9. App Log Analyser",
        "10. Legacy Code Convertor",
        "11. Equipment Predictive Maintenance",
        "12. Car Remote Diagnostics",
        "13. Auto OEM Market Research",
        "14. SDLC Multi-Agent",
        "15. Trade Negotiator Agent",
        "16. Automotive Campaigns Creation",
        "17. Supplier Negotiation System"
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
            abs_repo_path = os.path.abspath(repo_path)
            if not os.path.isdir(abs_repo_path):
                st.error(f"❌ The specified repo path does not exist: {abs_repo_path}")
                st.stop()
            if not os.path.isdir(os.path.join(abs_repo_path, ".git")):
                st.error(
                    f"❌ This folder is not a Git repository (missing .git): {abs_repo_path}"
                )
                st.stop()
            full_file_path = os.path.join(abs_repo_path, file_path)
            if not os.path.exists(full_file_path):
                if st.session_state.generated_code:
                    with open(full_file_path, "w", encoding="utf-8") as f:
                        f.write(st.session_state.generated_code)
                    st.info(f"📝 {file_path} was created from generated code.")
                else:
                    st.error(f"❌ File not found: {full_file_path}")
                    st.stop()
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
    logs = st.text_area(
        "Enter your logs here (App/Web and DB)...", value=st.session_state.ticket
    )
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
                if "Python" in target_language:
                    st.code(translated_code, language="python")
                elif "Java" in target_language:
                    st.code(translated_code, language="java")
                else:
                    st.code(translated_code, language="csharp")

# --- Predictive Maintenance Agent ---
elif step == "11. Equipment Predictive Maintenance":
    st.subheader("🏭 Predictive Maintenance Agent for Factory Floor")
    st.markdown("Upload sensor data from factory machinery to predict potential failures.")
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
    uploaded_file = st.file_uploader("Upload your sensor data (CSV)", type="csv")
    if uploaded_file is not None and llm:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("### Sensor Data Preview:")
            st.dataframe(df.head())
            data_string = df.to_string(index=False)
            maintenance_prompt = PromptTemplate.from_template(
                """
You are an expert Predictive Maintenance AI Agent for an automotive factory floor.
Your task is to analyze the following real-time sensor data from our machinery and provide a detailed maintenance report.
**Sensor Data:**

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
            if st.button("🤖 Analyze and Predict Failures"):
                with st.spinner("Analyzing sensor data and predicting outcomes..."):
                    chain = LLMChain(llm=llm, prompt=maintenance_prompt)
                    report = chain.run(sensor_data=data_string)
                    st.write("### Predictive Maintenance Report:")
                    st.markdown(report)
        except Exception as e:
            st.error(f"❌ An error occurred while processing the file: {e}")

# --- NEW: Remote Diagnostics & Service Booking Agent ---
elif step == "12. Car Remote Diagnostics":
    st.subheader("📡 Remote Diagnostics & Service Booking Agent")
    st.markdown("Analyze vehicle Diagnostic Trouble Codes (DTCs) to streamline the service process.")
    dtc_code = st.text_input("Enter the Diagnostic Trouble Code (DTC)", value="P0101")
    vehicle_model = st.selectbox(
        "Select Vehicle Model",
        (
            "Range Rover - Autobiography",
            "Range Rover Sport - Dynamic SE",
            "Range Rover Velar - Dynamic HSE",
            "Range Rover Evoque - S",
            "Discovery - Dynamic HSE",
            "Jaguar F-PACE - R-Dynamic SE",
        ),
    )
    if llm:
        diagnostics_prompt = PromptTemplate.from_template(
            """
You are an expert Automotive Remote Diagnostics AI Agent. Your goal is to analyze a Diagnostic Trouble Code (DTC) and provide a clear, two-part report.
**Vehicle Information:**
- **Model:** {vehicle_model}
- **DTC Code:** {dtc_code}
---
### **Part 1: Customer-Facing Report**
* **What is the issue?** Explain the problem in simple, non-technical language that a car owner can easily understand.
* **Severity Level:** Classify the severity on a scale: **Critical (Stop Driving Immediately)**, **High (Service Required Soon)**, **Medium (Monitor the Issue)**, or **Low (Informational)**.
* **Recommended Action for Driver:** Provide a clear, direct instruction for the driver (e.g., "Please pull over safely and call for roadside assistance.", "We recommend booking a service appointment within the next 3-5 business days.").
---
### **Part 2: Dealership Service Request**
* **Probable Cause:** Based on the DTC and vehicle model, list the most likely technical causes (e.g., "Faulty spark plug in cylinder 1," "Clogged catalytic converter," "Oxygen sensor malfunction").
* **Recommended Diagnostic Steps:** Outline the steps a technician should take to confirm the diagnosis.
* **Suggested Parts to Pre-order:** List any specific parts that the dealership should consider ordering in advance to expedite the repair (e.g., "Ignition Coil Pack (Part # 8C-2345)", "Upstream O2 Sensor (Part # 9A-1102)").
* **Recommended to buy:** Provide online link to buy the part.
"""
        )
        if st.button("🔍 Diagnose Vehicle Issue"):
            if not dtc_code:
                st.warning("Please enter a DTC code to analyze.")
            else:
                with st.spinner(f"Diagnosing DTC {dtc_code} for {vehicle_model}..."):
                    chain = LLMChain(llm=llm, prompt=diagnostics_prompt)
                    report = chain.run(vehicle_model=vehicle_model, dtc_code=dtc_code)
                    st.write("### Diagnostics & Service Report:")
                    st.markdown(report)

elif step == "13. Auto OEM Market Research":
    st.subheader("🚗🆚🚙 Auto OEM Market Research")
    st.markdown("Compare different car trims from leading brands based on features, cost, and customer reviews.")
    car_trims = [
        "BMW - 3 Series Sedan",
        "BMW - X5 SUV",
        "Mercedes-Benz - C-Class Sedan",
        "Mercedes-Benz - GLE SUV",
        "Range Rover - Evoque S",
        "Range Rover - Sport Dynamic SE",
        "Aston Martin - Vantage",
        "Aston Martin - DBX",
    ]
    trim1 = st.selectbox("Select Trim 1:", car_trims)
    trim2 = st.selectbox("Select Trim 2:", car_trims)
    if llm:
        comparison_prompt = PromptTemplate.from_template(
            """
You are an expert Automotive Market Research Agent. Your task is to provide a detailed comparison between two car trims.
Analyze and provide a tabular comparison of features (major and minor) including Price-to-Feature Ratio and Efficiency Rating (MPG/MPGe), cost, and customer reviews.
Use a table format with columns for "Aspect", "Details for {trim1}", and "Details for {trim2}".
For customer reviews, provide a star rating out of 5 and a brief summary.
**Comparison Request:**
- **Trim 1:** {trim1}
- **Trim 2:** {trim2}
---
### **Tabular Comparison**
| Aspect | Details for {trim1} | Details for {trim2} |
|---|---|---|
| **Major Features** | [List of major features] | [List of major features] |
| **Minor Features** | [List of minor features] | [List of minor features] |
| **Cost (MSRP)** | [Approximate MSRP] | [Approximate MSRP] |
| **Customer Reviews** | [Star Rating out of 5] | [Star Rating out of 5] |
| | [Brief Summary] | [Brief Summary] |
"""
        )
        if st.button("📈 Compare Trims"):
            if trim1 == trim2:
                st.warning("Please select two different trims to compare.")
            else:
                with st.spinner(f"Comparing {trim1} and {trim2}..."):
                    chain = LLMChain(llm=llm, prompt=comparison_prompt)
                    comparison_report = chain.run(trim1=trim1, trim2=trim2)
                    st.write("### 🚗 Comparison Report:")
                    st.markdown(comparison_report)

# --- NEW: SDLC Multi-Agent Workflow ---
elif step == "14. SDLC Multi-Agent":
    st.subheader("🚀 SDLC Multi-Agent Workflow")
    st.markdown("Automate key SDLC steps by orchestrating a team of AI agents.")
    st.subheader("1️⃣ Agent 1: Upload BRD Document")
    brd_file = st.file_uploader("Upload your BRD (PDF or TXT)", type=["pdf", "txt"])
    brd_content = ""
    if brd_file:
        if brd_file.type == "text/plain":
            brd_content = brd_file.read().decode("utf-8")
            st.success("✅ BRD document uploaded and read successfully.")
        else:
            st.warning("⚠️ Only plain text (.txt) files are supported at this time for direct reading.")
    if st.button("▶️ Run SDLC Agents"):
        if not brd_content:
            st.error("Please upload a BRD document to start the workflow.")
        elif not llm:
            st.error("LLM is not initialized. Please check your API key.")
        else:
            st.info("Starting the multi-agent SDLC workflow...")
            with st.spinner("2️⃣ Agent 2: Creating user stories from BRD..."):
                try:
                    user_stories = brd_to_user_stories(brd_content)
                    st.success("✅ User stories generated.")
                    st.subheader("📝 Generated User Stories:")
                    st.write(user_stories)
                    st.download_button(
                        label="⬇️ Download User Stories",
                        data=user_stories,
                        file_name="user_stories.txt",
                        mime="text/plain",
                    )
                except Exception as e:
                    st.error(f"❌ Error generating user stories: {e}")
                    st.stop()
            with st.spinner("3️⃣ Agent 3: Generating acceptance criteria..."):
                try:
                    acceptance_criteria = user_stories_to_acceptance_criteria(user_stories)
                    st.success("✅ Acceptance criteria generated.")
                    st.subheader("📋 Generated Acceptance Criteria:")
                    st.markdown(acceptance_criteria)
                    st.download_button(
                        label="⬇️ Download Acceptance Criteria",
                        data=acceptance_criteria,
                        file_name="acceptance_criteria.txt",
                        mime="text/plain",
                    )
                except Exception as e:
                    st.error(f"❌ Error generating acceptance criteria: {e}")
                    st.stop()

            with st.spinner("4️⃣ Agent 4: Writing the final code..."):
                try:
                    final_code = generate_code_from_requirements(user_stories, acceptance_criteria)
                    st.success("✅ Final code generated.")
                    st.subheader("💻 Generated Code:")
                    st.code(final_code, language="python")
                    st.download_button(
                        label="⬇️ Download Code",
                        data=final_code,
                        file_name="generated_code.py",
                        mime="text/plain",
                    )
                except Exception as e:
                    st.error(f"❌ Error generating final code: {e}")
                    st.stop()

            with st.spinner("5️⃣ Agent 5: Generating test cases..."):
                try:
                    test_cases = user_stories_to_test_cases(user_stories)
                    st.success("✅ Test cases generated.")
                    st.subheader("📋 Generated Test Cases:")
                    st.markdown(test_cases)
                    st.download_button(
                        label="⬇️ Download Test Cases",
                        data=test_cases,
                        file_name="test_cases.txt",
                        mime="text/plain",
                    )
                except Exception as e:
                    st.error(f"❌ Error generating test cases: {e}")
                    st.stop()
            st.balloons()
            st.success("🎉 SDLC Multi-Agent workflow completed successfully!")
            
elif step == "15. Trade Negotiator Agent":
    st.subheader("🌐 Trade Negotiator Agent")
    st.markdown("Analyze global tariff scenarios to find the best market entry strategy for UK-based car exports.")
    company_name = st.text_input("Enter your company name:", value="UK Auto Co.")
    car_model = st.text_input("Enter the car model to export:", value="Velar")
    target_market = st.selectbox(
        "Select Target Market:",
        (
            "USA",
            "Canada",
            "European Union (EU)",
            "Australia",
            "China",
            "Japan",
            "Brazil"
        )
    )
    tariff_rate = st.number_input(
        "Enter the current tariff rate for this market (in %):",
        min_value=0.0,
        max_value=100.0,
        value=2.5,
        step=0.1,
        format="%.1f"
    )
    base_price = st.number_input(
        "Enter the base price of the car (in £):",
        min_value=10000.0,
        value=50000.0,
        step=1000.0,
        format="%.2f"
    )
    market_data_input = st.text_area(
        "Provide any other relevant market data (regulations, demand trends, etc.):",
        value="""- Inflation Reduction Act (IRA) impact on EV tax credits.
- Strong demand for SUVs.
- High environmental standards."""
    )
    if llm:
        negotiator_prompt = PromptTemplate.from_template(
            """
You are an expert International Trade Negotiator and Market Analyst. Your task is to draft a strategic market entry and mitigation plan for a UK automotive company, {company_name}, exporting its {car_model} to the {target_market}.
**Current Situation Analysis:**
- **Product:** {car_model} (a luxury car, manufactured in the UK).
- **Target Market:** {target_market}
- **Tariff Details:** The current tariff on this vehicle is {tariff_rate}% under the existing trade agreement.
- **Market Data:**
{market_data_input}
- **Pricing Data:** The base price of the car is £{base_price}.
---
### **Strategic Recommendations:**
1.  **Tariff Circumvention/Mitigation:** Identify and analyze potential strategies to legally minimize or avoid the {tariff_rate}% tax. Consider the viability of the following:
    * **Rules of Origin**: Can we leverage components or manufacturing processes from countries with more favorable trade agreements to lower the effective tariff?
    * **FTAs**: Are there any existing or potential Free Trade Agreements (e.g., UK-USA) that could be leveraged?
    * **Reclassification**: Is it possible to reclassify the vehicle or its components to a lower-tariff category?
    * **In-country Investment**: What are the pros and cons of establishing a local assembly or finishing plant in the USA to qualify for domestic status or government incentives (e.g., Inflation Reduction Act benefits)?
2.  **Financial and Pricing Strategy:**
    * **Cost-Benefit Analysis**: Provide a high-level calculation showing the financial impact of the tariff on a single car with a base price of £{base_price}. Calculate the total landed cost with the tariff, and compare this to a scenario with a hypothetical 0% tariff.
    * **Pricing Models**: Analyze the pros and cons of three specific pricing models to address the tariff.
        1.  **Full Pass-Through**: Analyze the impact of passing the full {tariff_rate}% cost directly to the consumer. What are the risks to sales volume and brand perception?
        2.  **Cost Absorption**: Analyze the impact of absorbing the full {tariff_rate}% cost to maintain a competitive price. What is the impact on our profit margins?
        3.  **Hybrid Surcharge**: Analyze the impact of a transparent, separate tariff surcharge on the invoice. How can this be communicated to customers effectively?
3.  **Policy and Public Relations Approach:**
    * **Lobbying**: Suggest key government bodies or trade associations to engage with in the USA to advocate for a reduction or exemption of the tariff.
    * **Public Relations**: Recommend a public messaging strategy that frames our company's position on the tariffs.
4.  **Overall Commercial Team Briefing**: Summarize the key findings and provide a clear, prioritized list of three to five actionable steps for the commercial team to execute, including a recommended timeline (e.g., Short-term, Mid-term).
Please format the response as a clear, professional recommendation report with distinct sections. Do NOT use the term "memo".
"""
        )
        if st.button("📈 Generate Market Strategy Report"):
            if company_name and car_model and tariff_rate:
                with st.spinner(f"Analyzing market scenarios for {target_market}..."):
                    chain = LLMChain(llm=llm, prompt=negotiator_prompt)
                    report = chain.run(
                        company_name=company_name,
                        car_model=car_model,
                        target_market=target_market,
                        tariff_rate=tariff_rate,
                        base_price=base_price,
                        market_data_input=market_data_input
                    )
                    st.write("### 📈 Trade Strategy Report:")
                    st.markdown(report)
            else:
                st.warning("Please fill in all the details to generate the report.")

# --- NEW: Automotive Campaigns Creation Agent ---
elif step == "16. Automotive Campaigns Creation":
    st.subheader("🚀 Multi Agent Automated Automotive Campaign Creation")
    st.markdown("Automate the entire campaign process from research to content.")

    st.markdown("### 1. Provide Campaign Information")
    product_data = st.text_area(
        "Enter key vehicle features, price, and target buyer profile:",
        key="campaign_product_data"
    )
    competitor_data = st.text_area(
        "Enter key competitor details and market positioning:",
        key="campaign_competitor_data"
    )

    if st.button("▶️ Run Full Campaign Workflow"):
        if not product_data or not competitor_data:
            st.error("Please provide both product and competitor data to start.")
            st.stop()
        
        st.info("Starting the multi-agent campaign workflow...")

        # Agent 1: Market Research Agent
        with st.spinner("1/4: Analyzing market and generating strategy brief..."):
            try:
                market_brief = run_market_research_agent(llm, product_data, competitor_data)
                st.success("✅ Strategy brief generated.")
                st.markdown("### 📈 Strategic Brief")
                st.write(market_brief)
            except Exception as e:
                st.error(f"❌ Market Research Agent failed: {e}")
                st.stop()

        # Agent 2: Creative Strategy Agent
        with st.spinner("2/4: Developing creative concepts and messaging..."):
            try:
                creative_concept = run_creative_agent(llm, market_brief)
                st.success("✅ Creative concept developed.")
                st.markdown("### 🎨 Creative Concept")
                st.write(creative_concept)
            except Exception as e:
                st.error(f"❌ Creative Agent failed: {e}")
                st.stop()

        # Agent 3: Content Generation Agent
        with st.spinner("3/4: Generating ad copy and assets..."):
            try:
                digital_assets = run_content_agent(llm, creative_concept)
                st.success("✅ Digital assets generated.")
                st.markdown("### ✍️ Generated Content")
                st.write(digital_assets)
            except Exception as e:
                st.error(f"❌ Content Agent failed: {e}")
                st.stop()

        # Agent 4: Campaign Execution Agent
        with st.spinner("4/4: Generating deployable code and assets..."):
            try:
                deployable_code = run_execution_agent(llm, digital_assets)
                st.success("✅ Deployable code generated.")
                st.markdown("### 💻 Deployable Code")
                st.code(deployable_code, language="html")
            except Exception as e:
                st.error(f"❌ Execution Agent failed: {e}")
                st.stop()

        st.balloons()
        st.success("🎉 Campaign workflow completed successfully!")
        
# --- NEW: Supplier Negotiation System ---
elif step == "17. Supplier Negotiation System":
    st.subheader("🤝 Supplier Negotiation System")
    st.markdown("An AI agent system to help Sourcing Managers identify and negotiate with automotive part suppliers.")

    st.subheader("Step 1: Define Sourcing Request")
    part_name = st.text_input("Enter the Automotive Part Name:", value="G-Series Turbocharger")
    target_price = st.number_input(
        "Enter your Target Per-Unit Price (£):",
        min_value=1.0,
        value=1550.0,
        step=10.0,
        format="%.2f"
    )

    if 'supplier_research_done' not in st.session_state:
        st.session_state.supplier_research_done = False
    if 'supplier_options' not in st.session_state:
        st.session_state.supplier_options = []
    if 'supplier_profiles' not in st.session_state:
        st.session_state.supplier_profiles = {}


    if st.button("1. 🕵️ Find Top 3 Suppliers"):
        if not part_name:
            st.warning("Please enter a part name.")
        else:
            with st.spinner("Market Research Agent is identifying top suppliers..."):
                try:
                    research_results = run_supplier_research_agent(llm, part_name)
                    st.session_state.supplier_research_done = True
                    # A simple parser to extract names for the selectbox
                    profiles = research_results.split("###") # Assuming this is the separator
                    supplier_names = []
                    supplier_profiles_map = {}
                    for profile in profiles:
                        if "Supplier" in profile:
                            name = profile.split("\n")[0].replace("*","").strip()
                            supplier_names.append(name)
                            supplier_profiles_map[name] = profile.strip()

                    st.session_state.supplier_options = supplier_names
                    st.session_state.supplier_profiles = supplier_profiles_map
                    st.success("Research complete!")
                    st.markdown(research_results)

                except Exception as e:
                    st.error(f"An error occurred during market research: {e}")

    if st.session_state.supplier_research_done:
        st.subheader("Step 2: Select Supplier and Negotiate")
        if not st.session_state.supplier_options:
             st.warning("No suppliers found. Please try a different part name.")
        else:
            selected_supplier_name = st.selectbox(
                "Choose a supplier to negotiate with:",
                options=st.session_state.supplier_options
            )

            if st.button("2. 💬 Negotiate Price"):
                if not selected_supplier_name:
                    st.warning("Please select a supplier.")
                else:
                    supplier_profile = st.session_state.supplier_profiles.get(selected_supplier_name, "N/A")
                    with st.spinner(f"Negotiation Agent is engaging with {selected_supplier_name}..."):
                        try:
                            negotiation_result = run_negotiation_agent(llm, part_name, supplier_profile, target_price)
                            st.markdown("---")
                            st.subheader("Negotiation Outcome")
                            st.markdown(negotiation_result)
                        except Exception as e:
                            st.error(f"An error occurred during negotiation: {e}")





