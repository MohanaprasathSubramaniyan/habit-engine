# app/core/config.py
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    app_name: str = "Habit Formation Engine"
    version: str = "1.0.0"
    debug: bool = False

settings = Settings()