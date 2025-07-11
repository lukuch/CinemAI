from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    openai_api_key: str
    tmdb_api_key: str
    redis_url: str = "redis://localhost:6379/0"
    pgvector_url: str = "postgresql://user:password@localhost:5432/cinemai"
    class Config:
        env_file = ".env"

settings = Settings() 