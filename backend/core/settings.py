from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    openai_api_key: str = Field(validation_alias="OPENAI_API_KEY")
    tmdb_api_key: str = Field(validation_alias="TMDB_API_KEY")
    redis_url: str = Field(validation_alias="REDIS_URL")
    pgvector_url: str = Field(validation_alias="PGVECTOR_URL")

    class Config:
        env_file = ".env"
        extra = "allow"


settings = Settings()
