from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class Config(BaseSettings):

    app_root: Path = Path(__file__).resolve().parents[2]

    data_dir: Path | None = None
    models_dir: Path | None = None

    log_level: str = "INFO"
    log_dir: Path | None = None
    log_file: Path | None = None

    ohlcv_file_template: str = "{source}_{instrument}_{interval}_ohlcv.feather"
    
    hmm_file: Path | None = None
    
    parallel_workers: int | None = None

    model_config = SettingsConfigDict(
        env_file=".env", 
        case_sensitive=False,
    )

    def resolve_paths(self) -> None:
        root = Path(self.app_root)

        self.data_dir = self.data_dir or root / "data"
        self.log_dir = self.log_dir or root / "logs"
        self.models_dir = self.models_dir or root / "model_store"

        self.log_file = self.log_file or self.log_dir / "cRS.log"
        
        self.hmm_file = self.hmm_file or self.models_dir / "rs_hmm.joblib"
        
        self.parallel_workers = self.parallel_workers or 4
        
        for d in [self.data_dir, self.log_dir, self.models_dir]:
            d.mkdir(parents=True, exist_ok=True)

        for f in [self.log_file]:
            if not f.exists():
                f.touch()
        

