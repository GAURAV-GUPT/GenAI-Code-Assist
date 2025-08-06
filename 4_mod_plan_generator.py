import os
from dotenv import load_dotenv
# Load .env
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv


# Streamlit UI
st.set_page_config(page_title="🛠️ Code Modification Planner")
st.title("🛠️ GenAI: Code Change Plan Generator")

# User story input
ticket = st.text_area("Enter a user story or ticket (e.g. Add 2FA to login flow):")

if ticket:
    # Prompt for plan generation
    prompt = PromptTemplate.from_template(
        "Given this user story:\n\n{ticket}\n\nSuggest a detailed plan to modify the codebase. "
        "Mention which files/functions might change and what needs to be added or updated."
    )

    # LLM setup and execution
    llm = ChatOpenAI(model="gpt-4")
    chain = LLMChain(llm=llm, prompt=prompt)
    plan = chain.run(ticket=ticket)

    # Output
    st.subheader("📋 Suggested Modification Plan")
    st.write(plan)
