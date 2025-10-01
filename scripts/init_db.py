# backend/scripts/init_db.py
"""
Simple script to create DB tables. Useful for local SQLite demos.
"""
from app.database import engine
from app.models import Base

if __name__ == "__main__":
    print("Creating database tables...")
    Base.metadata.create_all(bind=engine)
    print("Done.")
