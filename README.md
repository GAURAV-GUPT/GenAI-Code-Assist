# Full GenAI Developer Assistant 🤖

This project is a multi-agent AI assistant designed to streamline various aspects of the software development lifecycle (SDLC) and other business operations. Built with **Streamlit** and **LangChain**, it integrates different AI-powered agents into a single, user-friendly interface.

🌟 Key Features
The application is structured around a collection of independent agents, each with a specific purpose. You can interact with them via the sidebar.

💻 Software Development Lifecycle (SDLC)
Ticket Summarization: Summarizes user stories and tickets, extracting key intent and tasks.

SDLC Multi-Agent: A chained workflow that takes a Business Requirements Document (BRD) and automatically generates:
User Stories
Acceptance Criteria
Test Cases

Functional Code

⚙️ **Industrial Automation & Diagnostics**

**CNC AI Agent:** This advanced workflow analyzes uploaded CNC machine data from a mock Excel file. It orchestrates three agents to provide:
An expert technical analysis of anomalies and root causes.
A step-by-step Standard Operating Procedure (SOP) for maintenance.
A business impact report quantifying the avoided downtime and cost savings.

**Observability to Self-Heal AI Agent:** Simulates an end-to-end "self-healing" IT process. It captures a mock system alert, creates an ITSM ticket, diagnoses the root cause, applies a fix, and closes the ticket—all autonomously.

**Litmus EDGE Agent:** Simulates an industrial diagnosis workflow for a CNC machine. It ingests live data, identifies equipment details, and provides a diagnostic report with a recommended solution.

**Car Remote Diagnostics:** A mock diagnostic agent that simulates troubleshooting and provides solutions for a car's Check Engine light.

**Equipment Predictive Maintenance:** A simple agent that provides a report on the health of a mock vehicle fleet.

📦 **Business & Supply Chain**
Supplier Negotiation System: Simulates a negotiation between a sourcing manager and a supplier for an automotive part.
Trade Negotiator Agent: A simple agent that provides a report on trade negotiation outcomes.
Accounts Payable Agent: Categorizes and summarizes a mock supplier query and drafts a professional response using provided "SAP" data.
Auto OEM Market Research: Generates a strategic brief, creative concept, and ad copy for a mock automotive campaign.

🛠️ **Technology Stack**
Framework: Streamlit for the front-end UI.
AI Orchestration: LangChain for chaining LLM calls and managing multi-agent workflows.
LLM: The system is powered by gpt-4o-mini via the OpenAI API.
Data Handling: Pandas for data manipulation

## 🚀 Features

The application provides a comprehensive suite of tools, including:

  * **Ticket Summarization**: Quickly summarize user stories or support tickets to extract key intent and tasks.
  * **Codebase Management**: Ingest and embed your codebase for semantic search and contextual understanding.
  * **Code Generation**: Generate or modify code based on a description of the required changes, using a few-shot prompting approach.
  * **Git Integration**: Automate the process of committing and pushing generated code to a Git repository.
  * **Log Analysis**: Analyze application and database logs to identify root causes of incidents and suggest improvements.
  * **Legacy Code Conversion**: Translate legacy code (e.g., Cobol, Fortran) to modern languages like Python or Java.
  * **Predictive Maintenance**: Analyze sensor data to predict equipment failures and recommend maintenance actions.
  * **Car Remote Diagnostics**: Decode and analyze vehicle diagnostic trouble codes (DTCs) to provide both a customer-facing report and a technical service request for a dealership.
  * **Auto OEM Market Research**: Compare different car trims based on features, cost, and customer reviews.
  * **SDLC Multi-Agent Workflow**: An orchestrated workflow that takes a Business Requirements Document (BRD) and automatically generates user stories, acceptance criteria, and test cases.

## 🛠️ Technologies Used

  * **Streamlit**: For building the interactive web application interface.
  * **LangChain**: The core framework for building the LLM-powered agents and managing prompt chains, embeddings, and vector stores.
  * **OpenAI**: The large language model (LLM) provider for all generative tasks (`gpt-4o-mini`).
  * **FAISS**: A library for efficient similarity search, used here for the codebase vector store.
  * **Python-Git**: For interacting with Git repositories to automate commits and pushes.
  * **Pandas**: For data handling, particularly for ingesting CSV files.
  * **python-dotenv**: To manage environment variables securely.

## ⚙️ Setup and Installation

### Prerequisites

1.  Python 3.8+
2.  An OpenAI API Key.

### Steps

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd <repository_folder>
    ```

2.  **Create and activate a virtual environment:**

    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install the required packages:**

    ```bash
    pip install -r requirements.txt
    ```

    *If a `requirements.txt` file is not present, you can create one from the code's imports:*

    ```bash
    pip install streamlit openai langchain gitpython pandas faiss-cpu python-dotenv
    ```

4.  **Set up your OpenAI API Key:**
    Create a `.env` file in the project root directory and add your key:

    ```
    OPENAI_API_KEY="your_api_key_here"
    ```

5.  **Run the Streamlit application:**

    ```bash
    streamlit run app.py
    ```

    *Note: Replace `app.py` with the name of your main script file if it's different.*

## 🤝 Contributing

Contributions are welcome\! If you have ideas for new agents or improvements to existing ones, feel free to open an issue or submit a pull request.
