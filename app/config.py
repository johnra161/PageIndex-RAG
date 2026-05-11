from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    openai_api_key: str = ""
    data_dir: Path = Path("data")
    max_pdf_size_mb: int = 1024

    @property
    def pdfs_dir(self) -> Path:
        return self.data_dir / "pdfs"

    @property
    def trees_dir(self) -> Path:
        return self.data_dir / "trees"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "jobs.db"

    def ensure_dirs(self):
        self.pdfs_dir.mkdir(parents=True, exist_ok=True)
        self.trees_dir.mkdir(parents=True, exist_ok=True)

    class Config:
        env_file = ".env"

settings = Settings()
settings.ensure_dirs()