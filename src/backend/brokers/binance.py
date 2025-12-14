import requests
from typing import List, Optional, Literal
import pandas as pd
from .base import BrokerAPI

class BinanceAPI(BrokerAPI):

    def __init__(self):
        self.session = requests.Session()
        self.perp_base = "https://fapi.binance.com/fapi/v1"
        self.spot_base = "https://api.binance.com/api/v3"

    def _safe_numeric(self, df: pd.DataFrame, exclude: List[str] = ["symbol"]):
        for col in df.columns:
            if col in exclude:
                continue
            try:
                df[col] = pd.to_numeric(df[col])
            except Exception:
                pass
        return df

    def get_symbols(self, 
                    instrument: Literal["spot", "perp"] = "spot"
                    ) -> List[str]:
        if instrument == "spot":
            url = f"{self.spot_base}/exchangeInfo"
        elif instrument == "perp":
            url = f"{self.perp_base}/exchangeInfo"
        resp = self.session.get(url, timeout=10)
        resp.raise_for_status()
        data = resp.json().get("symbols", [])
        
        return [item["symbol"] for item in data if 'symbol' in item]

    def get_klines(
        self,
        symbol: str,
        instrument: Literal["spot", "perp"] = "spot",
        interval: Literal["1m", "5m", "15m", "30m", "1h", "4h", "1d"] = "1d",
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time:   Optional[int] = None
    ) -> pd.DataFrame:
        
        if instrument == "spot":
            url = f"{self.spot_base}/klines"
        elif instrument == "perp":
            url = f"{self.perp_base}/klines"
            
        params = {"symbol": symbol, "interval": interval, "limit": limit}
        if start_time is not None:
            params["startTime"] = start_time
        if end_time is not None:
            params["endTime"] = end_time

        resp = self.session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        raw = resp.json()
        cols = [
            "time","open","high","low","close","volume",
            "closeTime","quoteAssetVolume","trades",
            "takerBuyBaseVolume","takerBuyQuoteVolume","ignore"
        ]
        df = pd.DataFrame(raw, columns=cols)
        df = df.drop(columns=["ignore","closeTime","quoteAssetVolume","trades","takerBuyBaseVolume","takerBuyQuoteVolume","ignore"], errors="ignore")
        df = self._safe_numeric(df, exclude=["time"])
        return df.sort_values("time").reset_index(drop=True)