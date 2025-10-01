# backend/app/config.py
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    DATABASE_URL: str = "sqlite:///./physician.db"  # default for local dev
    DEBUG: bool = True
    APP_NAME: str = "PhysicianNotetaker"

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()
