from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache


class Settings(BaseSettings):
    # Gemini API key (primary)
    google_api_key: str = ""

    # LangChain model names — both map to Gemini by default.
    # Swap critic_model to a heavier variant (e.g. gemini-1.5-pro) if needed.
    critic_model: str = "gemini-2.0-flash"   # full quality: critic + synthesis
    fast_model: str = "gemini-2.0-flash-lite" # cheap + fast: rubric builder + gap detector

    # Server
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False

    # Per-agent timeout in seconds
    agent_timeout: int = 30

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


@lru_cache()
def get_settings() -> Settings:
    return Settings()