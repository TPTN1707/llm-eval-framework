import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.database.connection import init_db

if __name__ == "__main__":
    print("--- Connecting to Neon.tech Cloud PostgreSQL and initializing tables ---")
    init_db()