from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = Field(env="OPENAI_API_KEY")
    tmdb_api_key: str = Field(env="TMDB_API_KEY")
    redis_url: str = Field(env="REDIS_URL")
    pgvector_url: str = Field(env="PGVECTOR_URL")

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()
