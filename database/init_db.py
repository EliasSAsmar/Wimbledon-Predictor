import os
import sqlite3
from pathlib import Path

def init_database():
    """Initialize the SQLite database with schema"""
    # Ensure database directory exists
    db_dir = Path(__file__).parent
    db_path = db_dir / 'tennis.db'
    
    # Read schema file
    schema_path = db_dir / 'schema.sql'
    with open(schema_path, 'r') as f:
        schema = f.read()
    
    # Connect to database and execute schema
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(schema)
        print(f"✅ Database initialized successfully at {db_path}")
    except Exception as e:
        print(f"❌ Error initializing database: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    init_database() 