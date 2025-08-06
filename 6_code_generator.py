import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from dotenv import load_dotenv
import os

# Load environment variables from .env
load_dotenv()
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# Streamlit page setup
st.set_page_config(page_title="💻 AI Code Generator")
st.title("💻 GenAI Code Modifier")

# Step 1: User pastes existing code or description
st.subheader("Step 1: Provide Existing Code Context")
context = st.text_area(
    "Paste the current function or describe the existing code behavior:",
    placeholder="def login(user):\n    return user.check_password()"
)

# Step 2: User describes what change is needed
st.subheader("Step 2: Describe the Change You Want to Make")
change = st.text_input(
    "E.g. Add OTP verification before password check",
    placeholder="Add OTP verification before password check"
)

# Step 3: Generate new code
if context and change and st.button("🚀 Generate Updated Code"):
    # Create prompt
    prompt = PromptTemplate.from_template(
        "Here is the existing code context:\n{context}\n\n"
        "You need to: {change}\n\n"
        "Please generate the updated Python code."
    )

    # Initialize LLM chain
    llm = ChatOpenAI(model="gpt-4")
    chain = LLMChain(llm=llm, prompt=prompt)

    # Get updated code
    with st.spinner("Generating code..."):
        updated_code = chain.run(context=context, change=change)

    # Output result
    st.success("✅ Here's your updated code:")
    st.code(updated_code, language="python")