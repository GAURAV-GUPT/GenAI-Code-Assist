# Full GenAI Developer Assistant 🤖

This project is a multi-agent AI assistant designed to streamline various aspects of the software development lifecycle (SDLC) and other business operations. Built with **Streamlit** and **LangChain**, it integrates different AI-powered agents into a single, user-friendly interface.

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
