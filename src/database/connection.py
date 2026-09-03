import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = os.getenv("DATABASE_URL")
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_NAME = os.getenv("DB_NAME")
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_PORT = os.getenv("DB_PORT", "5432")

def get_db_connection():
    """Establish and return a connection to the PostgreSQL database"""
    if DATABASE_URL:
        # Connect using the unified database URL
        return psycopg2.connect(DATABASE_URL)
    else:
        # Connect using individual connection credentials
        return psycopg2.connect(
            host=DB_HOST,
            database=DB_NAME,
            user=DB_USER,
            password=DB_PASSWORD,
            port=DB_PORT
        )

def init_db(schema_path="src/database/schema.sql"):
    """Read the SQL schema file and initialize all required database tables"""
    conn = None
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        
        # Read and execute the SQL file
        with open(schema_path, "r") as f:
            sql_script = f.read()
            
        cursor.execute(sql_script)
        conn.commit()
        print("Database tables initialized successfully.")
        
        cursor.close()
    except Exception as e:
        print(f"Error initializing database: {str(e)}")
    finally:
        if conn:
            conn.close()