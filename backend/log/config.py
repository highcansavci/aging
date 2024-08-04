from logging.config import dictConfig


class LogConfig:
    CONFIG = {
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "default": {
                "format": "[%(asctime)s] %(levelname)s in %(module)s: %(message)s",
            },
            "verbose": {
                "format": "[%(asctime)s] %(levelname)s [%(name)s:%(lineno)d] - %(message)s",
            },
        },
        "handlers": {
            "default": {
                "level": "INFO",
                "formatter": "default",
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
            },
            "file": {
                "level": "INFO",
                "formatter": "verbose",
                "class": "logging.FileHandler",
                "filename": "app.log",
            },
        },
        "loggers": {
            "": {"handlers": ["default", "file"], "level": "INFO"},
        },
    }


def setup_logging():
    dictConfig(LogConfig.CONFIG)
