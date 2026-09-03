
-- 1. Table to store reference Test Cases (Ground Truth Dataset)
CREATE TABLE IF NOT EXISTS test_cases (
    id SERIAL PRIMARY KEY,
    input_text TEXT NOT NULL,                  -- The input query/question to test
    expected_output TEXT,                      -- Optional reference answer (ground truth)
    context TEXT,                              -- Optional reference context/documents for RAG
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 2. Table to store metadata of each Evaluation Run (Benchmark Session)
CREATE TABLE IF NOT EXISTS evaluation_runs (
    id SERIAL PRIMARY KEY,
    run_name VARCHAR(255) NOT NULL,            -- e.g., 'GPT-4o Prompt v1 Test'
    model_name VARCHAR(100) NOT NULL,          -- e.g., 'gpt-4o-mini'
    prompt_template TEXT NOT NULL,             -- The system prompt template tested
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 3. Table to store detailed Evaluation Results of each test case per run
CREATE TABLE IF NOT EXISTS evaluation_results (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES evaluation_runs(id) ON DELETE CASCADE,
    test_case_id INTEGER REFERENCES test_cases(id) ON DELETE CASCADE,
    actual_output TEXT NOT NULL,               -- What the LLM generated under test
    latency_seconds FLOAT,                     -- Time taken to generate the response
    tokens_used INTEGER,                       -- Total tokens consumed (optional)
    
    -- Calculated evaluation metrics (standardized scale from 0.0 to 1.0)
    hallucination_score FLOAT,                 -- Score measuring hallucination rate
    answer_relevancy_score FLOAT,              -- Score measuring answer relevance
    faithfulness_score FLOAT,                  -- Score measuring truthfulness to context
    
    is_passed BOOLEAN DEFAULT TRUE,            -- Overall evaluation status based on thresholds
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);