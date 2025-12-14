from typing import Dict
from .base import BrokerAPI
from .mexc import MEXCAPI
from .bybit import BybitAPI
from .binance import BinanceAPI

__all__ = ["BrokerAPI", "MEXCAPI", "BybitAPI", "BinanceAPI"]

_registry: Dict[str, BrokerAPI] = {
    "mexc": MEXCAPI(),
    "bybit": BybitAPI(),
    "binance": BinanceAPI(),
}

def get_broker(name: str) -> BrokerAPI:
    try:
        return _registry[name]
    except KeyError:
        raise KeyError(f"Adapter '{name}' not found. Available: {list(_registry.keys())}")