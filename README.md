# LLM Evaluation & Benchmarking Framework 📊🧪

A production-grade, fully automated LLM Evaluation and Benchmarking Framework designed to quantitatively measure and monitor large language model outputs. 

Unlike simple API-calling tools, this framework serves as a robust test harness that evaluates LLM responses for **Answer Relevancy**, **Faithfulness**, and **Hallucination Rates** using state-of-the-art evaluation metrics, storing all structured run metadata and metrics in a cloud **PostgreSQL** database, and visualizing results on an interactive web **Streamlit Dashboard**.

To eliminate operational costs, the entire pipeline is configured to run on free-tier, high-speed **Groq Cloud API** models.

---

## 🚀 Key Features

- **Golden Dataset Management (PostgreSQL):** Stores reference questions, ground truth answers, and retrieval contexts in a cloud-hosted relational database.
- **Dynamic Response Generation (`src/generator/llm_client.py`):** Automatically invokes the target model under test (`qwen/qwen3.6-27b` reasoning model) via LangChain, tracking generation latency and token usage.
- **Custom LLM Evaluator (`src/evaluator/metrics.py`):** Forces Confident-AI's **DeepEval** framework to use a highly cost-effective, non-reasoning Groq model (`openai/gpt-oss-20b`) as the evaluation judge, eliminating 100% of OpenAI API costs.
- **Robust Thought-Stripping JSON Parser:** Features a custom brace-matching and regex-based parser that strips out model reasoning thoughts (`<think>...</think>`) and trailing conversational noise, delivering clean, parseable JSON payloads directly to DeepEval's internal validators.
- **Analytical Streamlit Dashboard (`dashboard.py`):** An elegant, single-page web interface providing metrics cards (Average Relevancy, Average Faithfulness, Average Latency, Pass Rate) and interactive detailed dataframes to analyze prompt engineering experiments.

---

## 📁 Project Directory Structure

    llm-eval-framework/
    ├── .env                         # API keys and Database URL (git-ignored)
    ├── .gitignore                   # Standard Git exclusion file
    ├── requirements.txt             # Project dependencies
    ├── uv.lock                      # Locked dependency versions (managed by uv)
    ├── main.py                      # Central pipeline orchestrator (Test Harness)
    ├── dashboard.py                 # Streamlit visualization dashboard
    │
    └── src/                         # Source directory
        ├── __init__.py
        ├── config.py                # Centralized configuration loader
        │
        ├── database/                # Database management module
        │   ├── __init__.py
        │   ├── connection.py        # Connection setup and initializers
        │   ├── schema.sql           # Database tables layout
        │   └── seed_db.py           # Seed script for initial test cases
        │
        ├── generator/               # LLM Generation module
        │   ├── __init__.py
        │   └── llm_client.py        # Target model inference engine
        │
        └── evaluator/               # LLM Evaluation module
            ├── __init__.py
            └── metrics.py           # Custom DeepEval LLM evaluator and metrics

## 🛠️ Tech Stack & Architecture

The llm-eval-framework project uses a highly optimized, modular architecture running on the following technology stack:

- **Frontend/Dashboard:** Streamlit (clean single-page interactive web UI with stretching dataframes).
- **LLM Generator & Evaluator (100% Groq-Powered):**
  - **Generator Client:** Uses LangChain with `qwen/qwen3.6-27b` (reasoning model) to generate factual outputs under latency and token tracking.
  - **Evaluator Client:** Forces Confident-AI's **DeepEval** to run on `openai/gpt-oss-20b` (standard model) via `DeepEvalBaseLLM` to evaluate metrics without OpenAI costs.
- **Database Backend:** Cloud PostgreSQL (Neon.tech) with structured tables managed by raw SQL schemas.

---

## 💻 Getting Started

### Prerequisites

You only need a single free API key and a database connection string:
1. A **Groq API Key** from the [Groq Console](https://console.groq.com/).
2. A **PostgreSQL Connection String** from [Neon.tech](https://neon.tech/).

### Installation

This project manages packages using Astral's fast Python package installer, `uv`.

1. Navigate to your project directory:

    cd llm-eval-framework

2. Create a `.env` file in the root directory and add your credentials:

    DATABASE_URL=your_postgresql_neon_connection_string_here
    GROQ_API_KEY=your_groq_api_key_here

3. Sync and install all required dependencies from the requirements file:

    uv pip install -r requirements.txt

### Database Initialization

To initialize the database tables on your Neon.tech cloud and seed initial test cases, run:

    uv run python src/database/init_db.py
    uv run python src/database/seed_db.py

### Running the Application

1. **To Execute the Evaluation Pipeline (Test Harness):**
   This reads test cases, gets LLM responses, runs DeepEval metrics, and records results in PostgreSQL:

    uv run python main.py

2. **To Start the Dashboard Web Interface:**
   This launches the interactive Streamlit dashboard to visualize results and compare metrics:

    uv run python -m streamlit run dashboard.py