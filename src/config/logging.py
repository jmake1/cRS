from logging.config import dictConfig

def setup_logging(config):

    dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "plain": {
                "format": "%(asctime)s %(levelname)s %(name)s :: %(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S",
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "formatter": "plain",
                "level": str(config.log_level),
            },
            "file": {
                "class": "logging.handlers.TimedRotatingFileHandler",
                "formatter": "plain",
                "level": "DEBUG",
                "filename": str(config.log_file),
                "when": "midnight",
                "backupCount": 14,
                "encoding": "utf-8",
                "utc": True,
            },
        },
        "root": {
            "level": "INFO",
            "handlers": ["console", "file"],
        },
    })
