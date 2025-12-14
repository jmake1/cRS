# brokers/bybit.py

import requests
from typing import List, Optional, Literal
import pandas as pd
from .base import BrokerAPI
import time

class BybitAPI(BrokerAPI):
    def __init__(self):
        self.session = requests.Session()
        self.BASE_URL = "https://api.bybit.com"
        
    def _ts_ms(self) -> int:
        return int(time.time() * 1000)

    def get_symbols(self, 
                    instrument: Literal["spot", "perp"] = "spot"
                    ) -> List[str]:

        url = f"{self.BASE_URL}/v5/market/tickers"
        
        if instrument == "spot":
            category = "spot"
        elif instrument == "perp":
            category = "linear"
            
        params = {"category": category}

        resp = self.session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [item['symbol'] for item in data["result"]["list"] if 'symbol' in item]

    def get_klines(
        self,
        symbol: str,
        instrument: Literal["spot", "perp"] = "spot",
        interval: Literal["1m", "5m", "15m", "30m", "1h", "4h", "1d"] = "1d",
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time:   Optional[int] = None
    ) -> pd.DataFrame:

        interval_map = {"1m": "1",
                        "5m": "5",
                        "15m": "15",
                        "30m": "30",
                        "1h": "60",
                        "4h": "240",
                        "1d": "D",
                        "1w": "W",
                        "1m": "M"}
        api_int = interval_map.get(interval.lower(), interval)

        url = f"{self.BASE_URL}/v5/market/kline"
        
        if instrument == "spot":
            category = "spot"
        elif instrument == "perp":
            category = "linear"
        
        params = {
            "category": category,
            "symbol":   symbol,
            "interval": api_int,
            "limit":    limit
        }
        
        if start_time is not None: params["start"] = start_time
        if end_time   is not None: params["end"]   = end_time
        
        resp   = self.session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        klines = resp.json()["result"]["list"]
        
        df = pd.DataFrame(klines, columns=["time","open","high","low","close","volume","turnover"])
        df = df[["time","open","high","low","close","volume","turnover"]].astype({
            "time":"int64","open":"float32","high":"float32",
            "low":"float32","close":"float32","volume":"float32","turnover":"float32"
        })

        df = df.drop(columns=['turnover']).sort_values("time").reset_index(drop=True)
        return df