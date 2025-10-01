import psycopg2
from config.config import DB_CONFIG

def get_connection():
    conn = psycopg2.connect(
        user=DB_CONFIG["user"],
        password=DB_CONFIG["password"],
        host=DB_CONFIG["host"],
        port=DB_CONFIG["port"],
        database=DB_CONFIG["database"]
    )
    return conn

def create_tables():
    """Create required tables if not exist"""
    queries = [
        """
        CREATE TABLE IF NOT EXISTS patients (
            patient_id SERIAL PRIMARY KEY,
            name VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS consultations (
            consultation_id SERIAL PRIMARY KEY,
            patient_id INT REFERENCES patients(patient_id) ON DELETE CASCADE,
            transcript TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS medical_summary (
            summary_id SERIAL PRIMARY KEY,
            consultation_id INT REFERENCES consultations(consultation_id) ON DELETE CASCADE,
            symptoms TEXT,
            diagnosis TEXT,
            treatment TEXT,
            current_status TEXT,
            prognosis TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS sentiment_analysis (
            sentiment_id SERIAL PRIMARY KEY,
            consultation_id INT REFERENCES consultations(consultation_id) ON DELETE CASCADE,
            sentiment VARCHAR(50),
            intent VARCHAR(100),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """,
        """
        CREATE TABLE IF NOT EXISTS soap_notes (
            note_id SERIAL PRIMARY KEY,
            consultation_id INT REFERENCES consultations(consultation_id) ON DELETE CASCADE,
            subjective TEXT,
            objective TEXT,
            assessment TEXT,
            plan TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        """
    ]

    conn = get_connection()
    cur = conn.cursor()
    for query in queries:
        cur.execute(query)
    conn.commit()
    cur.close()
    conn.close()
    print("✅ All tables created successfully")

if __name__ == "__main__":
    create_tables()
