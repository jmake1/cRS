import pandas as pd
import logging
import os
from datetime import datetime as _datetime
import joblib
from typing import Literal, List, Dict
from pathlib import Path

from config import config
from .kline_manager import update_klines

logger = logging.getLogger(__name__)

def save_model(model_type: Literal['garch', 'hmm'], 
               model: object, 
               meta: Dict
               ):
    meta.update({
        "saved_at": _datetime.utcnow().isoformat() + "Z",
    })
    path = Path(f"{config.models_dir}/{model_type}.joblib")
    logger.info(f"Saving new {model_type} model in {path}")
    payload = {"model": model, "meta": meta}
    tmp = path.with_suffix(".tmp")
    joblib.dump(payload, tmp, compress=("xz", 3))
    os.replace(tmp, path)
    logger.info(f"Saved new {model_type} model in {path}")
        
        
def load_model(model_type: Literal['garch', 'hmm']) -> tuple[object, Dict]:
    path = Path(f"{config.models_dir}/{model_type}.joblib")
    logger.info(f"Loading {model_type} model in {path}")
    if not path.exists():
        logger.info(f"Model {model_type} does not exist in {path}")
        return None, None
    obj = joblib.load(path)
    logger.info(f"Loaded {model_type} model in {path}")
    return obj.get("model", None), obj.get("meta", {})

def format_path(source: Literal['mexc', 'bybit', 'binance'], 
                instrument: Literal['spot', 'perp'], 
                interval: Literal["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
                ) -> str:
    filename = config.ohlcv_file_template.format(source=source, instrument=instrument, interval=interval)
    path = config.data_dir / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def update_ohlcv(sources: List[Literal['binance', 'mexc', 'bybit']], 
                instruments: List[Literal['spot', 'perp']], 
                interval: Literal["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
                ):
    for source in sources:
        for instrument in instruments:
            datafile = format_path(source, instrument, interval)
            update_klines(source=source, file_path=datafile, instrument=instrument, interval=interval)
    
def load_ohlcv(sources: List[Literal['binance', 'mexc', 'bybit']], 
                instruments: List[Literal['spot', 'perp']], 
                interval: Literal["1m", "5m", "15m", "30m", "1h", "4h", "1d"]
                ) -> pd.DataFrame:
    dfs = []
    for source in sources:
        for instrument in instruments:
            datafile = format_path(source, instrument, interval)
            df = pd.read_feather(datafile)
            df.sort_values(['datetime', 'symbol'], inplace=True)
            df['symbol'] = df['symbol'] + f"_{source}_{instrument}"
            dfs.append(df)
    final_df = pd.concat(dfs, axis=0)
    return final_df
    
def save_data_file(name: str, df: pd.DataFrame, format: str = 'feather'):
    if format == 'csv':
        path = Path(f"{config.data_dir}/{name}.csv")
        df.to_csv(path)
    else:
        path = Path(f"{config.data_dir}/{name}.feather")
        df.to_feather(path)
    logger.info(f"Saved {name} in {path}")
    
def load_data_file(name: str, format: str = 'feather'):
    if format == 'csv':
        path = Path(f"{config.data_dir}/{name}.csv")
        df = pd.read_csv(path)
    else:
        path = Path(f"{config.data_dir}/{name}.feather")
        df = pd.read_feather(path)
    logger.info(f"Loaded {name} from {path}")
    return df