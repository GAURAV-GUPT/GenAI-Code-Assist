# Full GenAI Developer Assistant (Steps 1 to 19 Integrated)
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
    prompt = Prompt
