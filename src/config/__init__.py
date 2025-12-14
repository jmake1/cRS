from .config import *
from .logging import setup_logging

config = Config()
config.resolve_paths()

setup_logging(config)