import os
import sys
import pandas as pd
import streamlit as st

sys.path.append(os.path.abspath(os.path.dirname(__file__)))
from src.database.connection import DATABASE_URL

# Page configuration
st.set_page_config(
    page_title="LLM Eval - Benchmark Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 LLM Evaluation & Benchmarking Dashboard")
st.write("Analyze, monitor, and compare LLM prompt performance and hallucination rates in real-time.")
st.write("---")

def load_evaluation_runs():
    """Fetch all completed evaluation runs from PostgreSQL using connection string to avoid Pandas warnings"""
    try:
        query = "SELECT id, run_name, model_name, created_at FROM evaluation_runs ORDER BY created_at DESC;"
        # Pass the DATABASE_URL connection string directly to silence DBAPI2 warnings
        df = pd.read_sql(query, DATABASE_URL)
        return df
    except Exception as e:
        st.error(f"Error loading evaluation runs: {str(e)}")
        return pd.DataFrame()

def load_evaluation_results(run_id):
    """Fetch detailed evaluation results for a specific run ID using connection string"""
    try:
        query = f"""
        SELECT 
            r.id as result_id,
            tc.input_text as "Question",
            tc.expected_output as "Ground Truth",
            r.actual_output as "LLM Output",
            r.latency_seconds as "Latency (s)",
            r.tokens_used as "Tokens",
            r.answer_relevancy_score as "Relevancy",
            r.faithfulness_score as "Faithfulness",
            r.hallucination_score as "Hallucination",
            r.is_passed as "Passed"
        FROM evaluation_results r
        JOIN test_cases tc ON r.test_case_id = tc.id
        WHERE r.run_id = {run_id}
        ORDER BY r.id ASC;
        """
        # Pass the DATABASE_URL connection string directly to silence DBAPI2 warnings
        df = pd.read_sql(query, DATABASE_URL)
        return df
    except Exception as e:
        st.error(f"Error loading evaluation results: {str(e)}")
        return pd.DataFrame()

# Load all available runs
runs_df = load_evaluation_runs()

if runs_df.empty:
    st.info("No evaluation runs found in the database. Please run main.py first to execute a benchmark.")
else:
    # Sidebar: Select Evaluation Run
    st.sidebar.title("🔍 Benchmark Selector")
    
    # Create a clean label for the dropdown selectbox
    run_options = {row['id']: f"{row['run_name']} ({row['model_name']})" for _, row in runs_df.iterrows()}
    selected_run_id = st.sidebar.selectbox(
        "Select Run Session:",
        options=list(run_options.keys()),
        format_func=lambda x: run_options[x]
    )
    
    # Fetch detailed run metadata for displaying in sidebar
    selected_run_info = runs_df[runs_df['id'] == selected_run_id].iloc[0]
    st.sidebar.write("---")
    st.sidebar.markdown(f"**Session Details:**")
    st.sidebar.write(f"- **ID:** `{selected_run_id}`")
    st.sidebar.write(f"- **Model:** `{selected_run_info['model_name']}`")
    st.sidebar.write(f"- **Created At:** `{selected_run_info['created_at']}`")
    
    # Load detailed results for the selected run
    results_df = load_evaluation_results(selected_run_id)
    
    if not results_df.empty:
        # Calculate summary metrics
        avg_relevancy = results_df["Relevancy"].mean()
        avg_faithfulness = results_df["Faithfulness"].mean()
        avg_latency = results_df["Latency (s)"].mean()
        total_tokens = results_df["Tokens"].sum()
        
        pass_rate = (results_df["Passed"].sum() / len(results_df)) * 100
        
        # Display Metric Cards
        st.markdown("### 📈 Session Summary Metrics")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        
        m_col1.metric(
            label="Average Answer Relevancy",
            value=f"{avg_relevancy:.4f}",
            help="Measures how relevant the generated answers are to the input questions."
        )
        m_col2.metric(
            label="Average Faithfulness (No Hallucination)",
            value=f"{avg_faithfulness:.4f}",
            help="Measures how truthful the generated answers are to the provided retrieval contexts."
        )
        m_col3.metric(
            label="Average Latency",
            value=f"{avg_latency:.2f} seconds",
            help="Measures average time taken to generate a response."
        )
        m_col4.metric(
            label="Overall Pass Rate",
            value=f"{pass_rate:.1f}%",
            help="Percentage of test cases that passed the defined threshold."
        )
        st.write("---")
        
        st.markdown("### 📋 Detailed Test Case Results")
        display_df = results_df.copy()
        display_df["Passed"] = display_df["Passed"].apply(lambda x: "✅ PASS" if x else "❌ FAIL")
        
        st.dataframe(
            display_df[[
                "Question", "Ground Truth", "LLM Output", 
                "Latency (s)", "Tokens", "Relevancy", "Faithfulness", "Passed"
            ]],
            width='stretch'
        )
        
    else:
        st.warning("No detailed results found for this run session.")