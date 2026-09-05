import os
import sys
import time
from dotenv import load_dotenv

sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.database.connection import get_db_connection
from src.generator.llm_client import generate_llm_response
from src.evaluator.metrics import evaluate_llm_output

load_dotenv()

def run_evaluation_pipeline(run_name="Qwen-27b Benchmark Run"):
    """
    Orchestrates the entire LLM evaluation pipeline:
    1. Fetches test cases from PostgreSQL.
    2. Generates actual responses from the model under test (Qwen 3.6).
    3. Evaluates the outputs using DeepEval metrics (GPT-OSS-20b on Groq).
    4. Saves all detailed logs and scores back to PostgreSQL.
    """
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # 1. Fetch all gold test cases from the database
        cursor.execute("SELECT id, input_text, expected_output, context FROM test_cases;")
        test_cases = cursor.fetchall()
        
        if not test_cases:
            print("Error: No test cases found in the database. Please run seed_db.py first.")
            return
            
        print(f"\n--- Starting Evaluation Pipeline: {run_name} ---")
        print(f"Found {len(test_cases)} test cases to evaluate.\n")
        
        # Define the system prompt for the model under test (Qwen 3.6)
        system_prompt = "You are a professional assistant. Answer the user's question accurately and concisely based on facts."
        model_under_test = "qwen/qwen3.6-27b"
        
        # 2. Register the new Evaluation Run in the database
        insert_run_query = """
        INSERT INTO evaluation_runs (run_name, model_name, prompt_template)
        VALUES (%s, %s, %s) RETURNING id;
        """
        cursor.execute(insert_run_query, (run_name, model_under_test, system_prompt))
        run_id = cursor.fetchone()[0]
        conn.commit()
        
        # 3. Iterate through each test case and execute generation + evaluation
        for idx, (tc_id, input_text, expected, context) in enumerate(test_cases, 1):
            print(f"[{idx}/{len(test_cases)}] Evaluating Test Case ID {tc_id}...")
            print(f" Query: {input_text}")
            
            # Step 3A: Generate response using the target model
            gen_result = generate_llm_response(
                system_prompt=system_prompt,
                user_input=input_text,
                model_name=model_under_test
            )
            actual_output = gen_result["output_text"]
            latency = gen_result["latency"]
            tokens = gen_result["tokens_used"]
            
            print(f" Generated Output: {actual_output}")
            print(f" Generation Latency: {latency:.2f}s | Tokens: {tokens}")
            
            # Step 3B: Evaluate the output using DeepEval metrics
            eval_result = evaluate_llm_output(
                input_text=input_text,
                actual_output=actual_output,
                expected_output=expected,
                context=context
            )
            
            relevancy = eval_result["relevancy_score"]
            faithfulness = eval_result["faithfulness_score"]
            is_passed = eval_result["is_passed"]
            
            print(f" Scores -> Relevancy: {relevancy:.4f} | Faithfulness: {faithfulness:.4f} | Passed: {is_passed}")
            
            # Step 3C: Save the detailed metrics back into PostgreSQL database
            insert_result_query = """
            INSERT INTO evaluation_results (
                run_id, test_case_id, actual_output, latency_seconds, tokens_used,
                hallucination_score, answer_relevancy_score, faithfulness_score, is_passed
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s);
            """
            # Note: DeepEval's faithfulness measures truthfulness, so hallucination_score is (1.0 - faithfulness)
            hallucination_score = 1.0 - faithfulness
            
            cursor.execute(insert_result_query, (
                run_id, tc_id, actual_output, latency, tokens,
                hallucination_score, relevancy, faithfulness, is_passed
            ))
            conn.commit()
            print(" Saved results to database.\n")
            
        print("====================================================")
        print(f"Pipeline Execution Complete for Run ID: {run_id}")
        print("All benchmark metrics successfully recorded in PostgreSQL.")
        print("====================================================")
        
        cursor.close()
    except Exception as e:
        print(f"Pipeline Error: {str(e)}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    run_evaluation_pipeline()