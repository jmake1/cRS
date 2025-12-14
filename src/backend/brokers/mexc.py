import requests
from typing import List, Optional, Literal
import pandas as pd
from .base import BrokerAPI

class MEXCAPI(BrokerAPI):

    def __init__(self):
        self.session = requests.Session()
        self.SPOT_BASE_URL = "https://api.mexc.com/api/v3"
        self.FUTURES_BASE_URL = "https://contract.mexc.com/api/v1/contract"

    def get_symbols(self, 
                    instrument: Literal["spot", "perp"] = "spot"
                    ) -> List[str]:
        
        if instrument == "spot":
            url = f"{self.SPOT_BASE_URL}/exchangeInfo"
        elif instrument == "perp":
            url = f"{self.FUTURES_BASE_URL}/detail"

        resp = self.session.get(url, timeout=10)
        resp.raise_for_status()
        raw = resp.json()
        if instrument == "spot":
            return [item['symbol'] for item in raw['symbols'] if 'symbol' in item]
        else:
            return [item['symbol'] for item in raw['data'] if 'symbol' in item]

    def get_klines(
        self,
        symbol: str,
        instrument: Literal["spot", "perp"] = "spot",
        interval: Literal["1m", "5m", "15m", "30m", "1h", "4h", "1d"] = "1d",
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time:   Optional[int] = None
    ) -> pd.DataFrame:
        
        interval_map = {
            '1m': 'Min1',
            '5m': 'Min5',
            '15m': 'Min15',
            '30m': 'Min30',
            '1h': 'Min60',
            '4h': 'Hour4',
            '1d': 'Day1',
            '1w': 'Week1',
            '1M': 'Month1',
        }
        
        if instrument == "spot":
            url = f"{self.SPOT_BASE_URL}/klines"
            if interval == "1h":
                int = "60m"
            else:
                int = interval
        elif instrument == "perp":
            url = f"{self.FUTURES_BASE_URL}/kline/{symbol}"
            int = interval_map.get(interval)
        
        params = {"interval": int, "limit": limit}
        if instrument == "spot":
            params["symbol"] = symbol
            if start_time is not None:
                params["startTime"] = start_time
            if end_time is not None:
                params["endTime"] = end_time
        elif instrument == "perp":
            if start_time is not None:
                params["start"] = start_time
            if end_time is not None:
                params["end"] = end_time
            
        resp = self.session.get(url, params=params, timeout=10)
        resp.raise_for_status()
        raw = resp.json()

        if instrument == "perp":
            if isinstance(raw, dict) and 'data' in raw and isinstance(raw['data'], dict):
                data_dict = raw['data']
            elif isinstance(raw, dict) and all(k in raw for k in ['time', 'open', 'high', 'low', 'close', 'vol']):
                data_dict = raw
            else:
                raise ValueError(f"Unexpected response structure for futures klines: {raw}")

            df = pd.DataFrame({
                'time': [t * 1000 for t in data_dict['time']],  # convert seconds to ms
                'open': data_dict['open'],
                'high': data_dict['high'],
                'low': data_dict['low'],
                'close': data_dict['close'],
                'volume': data_dict['vol'],
            })
            df = df.astype({
                'time': 'int64',
                'open': 'float32',
                'high': 'float32',
                'low': 'float32',
                'close': 'float32',
                'volume': 'float32'
            })
        elif instrument == "spot":
            columns = ['time', 'open', 'high', 'low', 'close', 'volume']
            df_data = [row[:6] for row in raw]
            df = pd.DataFrame(df_data, columns=columns)
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            numeric_columns = ['open', 'high', 'low', 'close', 'volume']
            df[numeric_columns] = df[numeric_columns].astype(float)
        return df.sort_values("time").reset_index(drop=True)