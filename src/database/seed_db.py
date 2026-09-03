import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.database.connection import get_db_connection

def seed_test_cases():
    """Insert initial benchmark test cases into the database"""
    # Sample golden dataset (Question, Expected Answer, Reference Context)
    test_cases = [
        (
            "What is the capital of France?", 
            "Paris", 
            "France is a country in Europe. Its capital city is Paris, which is known for its culture and history."
        ),
        (
            "Who won the FIFA World Cup in 2022?", 
            "Argentina", 
            "The 2022 FIFA World Cup was held in Qatar, and Argentina won the tournament after defeating France in the final."
        ),
        (
            "Explain quantum computing in one sentence.", 
            "Quantum computing uses quantum mechanics to process complex data much faster than classical computers.", 
            "Quantum computing is a rapidly-emerging technology that harnesses the laws of quantum mechanics to solve problems too complex for classical computers."
        )
    ]
    
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Check if test cases already exist to prevent duplicate seeding
        cursor.execute("SELECT COUNT(*) FROM test_cases;")
        count = cursor.fetchone()[0]
        
        if count == 0:
            print("Seeding initial benchmark test cases into the database...")
            insert_query = """
            INSERT INTO test_cases (input_text, expected_output, context)
            VALUES (%s, %s, %s);
            """
            cursor.executemany(insert_query, test_cases)
            conn.commit()
            print(f"Successfully seeded {len(test_cases)} test cases.")
        else:
            print(f"Database already contains {count} test cases. Skipping seeding step.")
            
        cursor.close()
    except Exception as e:
        print(f"Error seeding database: {str(e)}")
    finally:
        if conn:
            conn.close()

if __name__ == "__main__":
    seed_test_cases()