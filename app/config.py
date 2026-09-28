from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "embedding-service"
    app_env: str = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8081

    embedding_model_name: str = "BAAI/bge-m3"
    embedding_normalize: bool = True
    embedding_device: str = "cpu"
    embedding_max_batch_size: int = 64
    embedding_max_text_length: int = 8000

    # Optional: cap torch CPU threads to avoid oversubscription when several
    # requests run concurrently. 0 / None means "let torch decide".
    embedding_num_threads: int = 0

    # Max inference calls allowed to run at once. torch already uses every core
    # per encode, so the safe default is 1 (serialize) to avoid core
    # oversubscription/thrashing under concurrent load. Raise this only together
    # with a matching EMBEDDING_NUM_THREADS (cores / concurrency).
    embedding_max_concurrency: int = 1

    # Optional internal API key. Empty means auth is disabled.
    api_key: str | None = None

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",
    )


settings = Settings()
