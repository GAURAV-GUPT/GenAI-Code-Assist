# Full GenAI Developer Assistant (Steps 1 to 8 Integrated - Final Fixed)
import os

os.environ[
    "GIT_PYTHON_GIT_EXECUTABLE"
] = r"C:\Program Files\Git\cmd\git.exe"  # adjust path as needed

import git
import streamlit as st
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

step = st.sidebar.radio(
    "Available Agents:",
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
    ],
)

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
    if ticket:
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
    if file:
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
    if ticket:
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
        st.warning("Upload codebase first")

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
    if context and change:
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
    if code:
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
    logs = st.text_area("Enter your app logs here...", value=st.session_state.ticket)
    if logs:
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

# elif step == "10. Legacy Code Convertor":
