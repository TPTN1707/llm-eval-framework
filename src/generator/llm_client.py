import sys
import os
import time
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY is not set in the environment/.env file.")

def generate_llm_response(system_prompt, user_input, model_name="gpt-4o-mini", temperature=0.0):
    """
    Call OpenAI LLM using LangChain to generate a response.
    Measures generation latency and tracks token usage.
    """
    start_time = time.time()
    
    try:
        # Initialize the LangChain ChatOpenAI client
        llm = ChatOpenAI(
            model=model_name,
            temperature=temperature,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
        
        # Construct the prompt template (system and user roles)
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("user", "{input}")
        ])
        
        # Chain the prompt template and the LLM together
        chain = prompt_template | llm
        
        # Invoke the model with the user input
        response = chain.invoke({"input": user_input})
        
        # Calculate latency
        latency = time.time() - start_time
        
        # Extract token usage safely from response metadata
        tokens_used = 0
        metadata = response.response_metadata
        if "token_usage" in metadata:
            tokens_used = metadata["token_usage"].get("total_tokens", 0)
            
        return {
            "output_text": response.content,
            "latency": latency,
            "tokens_used": tokens_used
        }
        
    except Exception as e:
        print(f"Error calling LLM: {str(e)}")
        return {
            "output_text": f"Error: {str(e)}",
            "latency": 0.0,
            "tokens_used": 0
        }

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    
    sys_prompt = "You are a helpful and extremely concise assistant."
    user_q = "What is the capital of France?"
    print("--- Testing LLM Generator Client ---")
    result = generate_llm_response(sys_prompt, user_q)
    print(f"Output : {result['output_text']}")
    print(f"Latency: {result['latency']:.4f} seconds")
    print(f"Tokens : {result['tokens_used']}")