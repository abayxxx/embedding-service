from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "embedding-service"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8081

    embedding_model_name: str = "BAAI/bge-m3"
    # Hugging Face revision (commit hash) to load; empty = latest.
    embedding_model_revision: str | None = None
    embedding_normalize: bool = True
    embedding_device: str = "cpu"
    embedding_max_batch_size: int = 64
    embedding_max_text_length: int = 8000

    # Optional: cap torch CPU threads to avoid oversubscription when several
    # requests run concurrently. 0 / None means "let torch decide".
    embedding_num_threads: int = 0

    # Optional internal API key. Empty means auth is disabled.
    api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
