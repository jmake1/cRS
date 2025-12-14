from abc import ABC
from typing import List, Optional, Literal
import pandas as pd

class BrokerAPI(ABC):

    def get_symbols(self, 
                    instrument: Literal["spot", "perp"] = "spot"
                    ) -> List[str]:

        raise NotImplementedError("Symbols history not implemented for this broker")

    def get_klines(
        self,
        symbol: str,
        instrument: Literal["spot", "perp"] = "spot",
        interval: Literal["1m", "5m", "15m", "30m", "1h", "4h", "1d"] = "1d",
        limit: int = 1000,
        start_time: Optional[int] = None,
        end_time:   Optional[int] = None
    ) -> pd.DataFrame:

        raise NotImplementedError("Klines history not implemented for this broker") 