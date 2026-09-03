import sys
import os
import time
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

groq_key = os.getenv("GROQ_API_KEY")
if not groq_key:
    print("Warning: GROQ_API_KEY is not set in the environment/.env file.")

def generate_llm_response(system_prompt, user_input, model_name="groq/compound-mini", temperature=0.0):
    """
    Call Groq LLM using LangChain to generate a response.
    Measures generation latency and tracks token usage.
    Lists available models if 404 occurs.
    """
    start_time = time.time()
    
    try:
        # Initialize the LangChain ChatGroq client using the API key from .env
        llm = ChatGroq(
            model=model_name,
            temperature=temperature,
            api_key=groq_key
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
        
        # Extract token usage safely from Groq/OpenAI metadata format
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
        error_msg = str(e)
        if "404" in error_msg or "model_not_found" in error_msg:
            print(f"\n[Diagnostic] Model '{model_name}' failed. Listing your available Groq models:")
            try:
                from groq import Groq
                client = Groq(api_key=groq_key)
                models = client.models.list()
                for m in models.data:
                    print(f" - {m.id}")
            except Exception as list_err:
                print(f"Could not list Groq models: {str(list_err)}")
                
        return {
            "output_text": f"Error: {error_msg}",
            "latency": 0.0,
            "tokens_used": 0
        }

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    
    sys_prompt = "You are a helpful and extremely concise assistant."
    user_q = "What is the capital of France?"
    print("--- Testing LLM Generator Client using GROQ ---")
    result = generate_llm_response(sys_prompt, user_q)
    print(f"Output : {result['output_text']}")
    print(f"Latency: {result['latency']:.4f} seconds")
    print(f"Tokens : {result['tokens_used']}")