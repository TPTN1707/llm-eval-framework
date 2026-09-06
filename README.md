# LLM Evaluation & Benchmarking Framework 📊🧪

An automated, production-grade evaluation and benchmarking framework designed to quantitatively measure and monitor Large Language Model (LLM) outputs. 

Most developers only know how to build basic LLM applications, but this repository serves as a robust **test harness** that scores model performance for **Answer Relevancy**, **Faithfulness**, and **Hallucination Rates**. It stores all structured run metadata, latencies, token counts, and metrics in a cloud **PostgreSQL** database, and visualizes them on an interactive web **Streamlit Dashboard**.

To eliminate operational costs, the entire framework is configured to run on free-tier, high-speed **Groq Cloud API** models.

---

## 🏫 The Architecture Analogy: "The Automated Grading System"

To easily understand how the different modules of this framework interact, we can think of it as an **Automated School Exam System**:

    [Test Case DB (Exam Paper)] 
                 │
                 ▼ (input query)
     [LLM under Test (Student)] ───────► (calculates latency & tokens)
                 │
                 ▼ (actual answer)
    [DeepEval Metrics (Teacher)] ◄───── [Reference Context (Textbook)]
                 │
                 ▼ (evaluates Relevancy & Faithfulness)
     [PostgreSQL (Grade Book)]
                 │
                 ▼ (persists logs & scores)
      [Streamlit Dashboard]

- **The Exam Paper (`test_cases` table):** This is the gold benchmark dataset. It contains the testing questions (`input_text`), correct reference answers (`expected_output`), and the source textbooks (`context`).
- **The Student under Test (`llm_client.py`):** This is the target LLM being benchmarked (configured with the reasoning model `qwen/qwen3.6-27b`). It reads the questions and writes its responses while we track how fast the student answered (`latency_seconds`) and how many resources they consumed (`tokens_used`).
- **The Grading Teacher (`metrics.py`):** This is the custom `openai/gpt-oss-20b` evaluator model wrapped inside Confident-AI's **DeepEval** framework. The teacher grades the student's work on two strict criteria:
  - *Answer Relevancy (Score 0.0 - 1.0):* Did the student answer the question directly, or did they write redundant, wordy information?
  - *Faithfulness / Hallucination Check (Score 0.0 - 1.0):* Did the student copy truthfully from the provided context (textbook), or did they make up facts (hallucinate)?
- **The Grade Book (PostgreSQL Cloud):** Saves all the student's answers, latencies, and grades permanently in the cloud.
- **The Report Card (`dashboard.py`):** The final visual dashboard where you can view average scores, pass rates, and performance trends of different test runs.

---

## 🚀 Key Features

- **100% Free-Tier Operation:** Orchestrated entirely on Groq's LPU hardware, avoiding expensive OpenAI API usage while maintaining production-grade evaluation capabilities.
- **Dynamic Response Generation:** Integrates LangChain with advanced reasoning models to capture step-by-step thinking processes, complete with latency tracking and token consumption.
- **Robust JSON thought-stripping parser:** Features a custom brace-matching and regex-based parser that strips out model reasoning thoughts (`<think>...</think>`) and trailing conversational noise, delivering clean, parseable JSON payloads directly to DeepEval's internal validators.
- **Visual Performance Dashboard:** A streamlined Streamlit app with clean summary cards, detailed interactive tables, and automatic future-proof container widths.

---

## 📁 Project Directory Structure

    llm-eval-framework/
    ├── .env                         # Database URL and API keys (git-ignored)
    ├── .gitignore                   # Standard Git exclusion file
    ├── requirements.txt             # Project python dependencies
    ├── uv.lock                      # Locked dependency versions
    ├── main.py                      # Central pipeline orchestrator (Test Harness)
    ├── dashboard.py                 # Streamlit visualization dashboard
    │
    └── src/                         # Source directory
        ├── __init__.py
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

---

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

This project manages packages using `uv`.

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