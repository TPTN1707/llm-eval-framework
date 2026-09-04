import os
import sys
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from deepeval.models.base_model import DeepEvalBaseModel
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
from deepeval.test_case import LLMTestCase


load_dotenv()

# 1. Custom LLM Wrapper to force DeepEval to use our free Groq model as the evaluator
class GroqEvaluator(DeepEvalBaseModel):
    def __init__(self, model_name="qwen/qwen3.6-27b"):
        self.model_name = model_name
        # Initialize the LangChain ChatGroq client
        self.chat_model = ChatGroq(
            model=self.model_name,
            temperature=0.0, # Temperature 0 is ideal for strict evaluation logic
            api_key=os.getenv("GROQ_API_KEY")
        )

    def load_model(self):
        return self.chat_model

    def generate(self, prompt: str) -> str:
        """Synchronously generate evaluation response"""
        chat_model = self.load_model()
        return chat_model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        """Asynchronously generate evaluation response"""
        chat_model = self.load_model()
        res = await chat_model.ainvoke(prompt)
        return res.content

    def get_model_name(self):
        return self.model_name


def evaluate_llm_output(input_text, actual_output, expected_output=None, context=None):
    """
    Evaluate the LLM output using DeepEval metrics with our custom Groq evaluator.
    Returns calculated scores for Relevancy and Faithfulness.
    """
    try:
        # Initialize our custom Groq evaluator
        evaluator_model = GroqEvaluator()
        
        # Define the test case using DeepEval structure
        retrieval_context = [context] if context else None
        
        test_case = LLMTestCase(
            input=input_text,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=retrieval_context
        )
        
        # Initialize DeepEval metrics with our custom Groq model
        relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model=evaluator_model, async_mode=False)
        
        # Faithfulness metric requires context to evaluate hallucinations
        faithfulness_metric = None
        if retrieval_context:
            faithfulness_metric = FaithfulnessMetric(threshold=0.5, model=evaluator_model, async_mode=False)
            
        # Execute Relevancy evaluation
        print("Calculating Answer Relevancy...")
        relevancy_metric.measure(test_case)
        relevancy_score = relevancy_metric.score
        
        # Execute Faithfulness (Hallucination check) if context exists
        faithfulness_score = 1.0 # Default perfect score if no context to hallucinate from
        if faithfulness_metric:
            print("Calculating Faithfulness (Hallucination check)...")
            faithfulness_metric.measure(test_case)
            faithfulness_score = faithfulness_metric.score
            
        # Calculate overall pass status (e.g., passed if both metrics are >= 0.5)
        is_passed = relevancy_score >= 0.5 and faithfulness_score >= 0.5
        
        return {
            "relevancy_score": float(relevancy_score),
            "faithfulness_score": float(faithfulness_score),
            "is_passed": bool(is_passed)
        }
        
    except Exception as e:
        print(f"Error during evaluation: {str(e)}")
        return {
            "relevancy_score": 0.0,
            "faithfulness_score": 0.0,
            "is_passed": False
        }

if __name__ == "__main__":
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    
    test_q = "What is the capital of France?"
    test_output = "The capital of France is Paris."
    test_expected = "Paris"
    test_context = "France is a European country and its capital is Paris."
    
    print("--- Testing Custom Evaluator ---")
    results = evaluate_llm_output(test_q, test_output, test_expected, test_context)
    print(f"Relevancy Score: {results['relevancy_score']:.4f}")
    print(f"Faithfulness Score: {results['faithfulness_score']:.4f}")
    print(f"Is Passed      : {results['is_passed']}")