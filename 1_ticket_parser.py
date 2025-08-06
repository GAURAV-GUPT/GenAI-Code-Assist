from dotenv import load_dotenv
import os

load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate


llm = ChatOpenAI(model="gpt-4")

st.title("📝 Ticket Summarizer")
ticket = st.text_area("Enter a ticket or user story")

if ticket:
    prompt = PromptTemplate.from_template(
        "Summarize the following ticket and extract intent, task type, and key keywords:\n\n{ticket}"
    )
    chain = LLMChain(llm=llm, prompt=prompt)
    summary = chain.run(ticket=ticket)
    st.write("📌 Summary:")
    st.write(summary)
