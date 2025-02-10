"""Loggers configuration."""

import json
import logging

LOG_LEVEL = logging.INFO


class CustomHandler(logging.StreamHandler):
    """Custom handler for logging algorithm. Credits to Qibocal."""

    def __init__(self):
        """Initialize custom handler."""
        super().__init__()
        self.FORMATS = None

    def format(self, record):
        """Format the record with specific format."""
        fmt = "[%(levelname)s|%(asctime)s]: %(message)s"

        grey = "\x1b[38;20m"
        green = "\x1b[92m"
        yellow = "\x1b[33;20m"
        red = "\x1b[31;20m"
        bold_red = "\x1b[31;1m"
        reset = "\x1b[0m"

        self.FORMATS = {
            logging.DEBUG: green + fmt + reset,
            logging.INFO: grey + fmt + reset,
            logging.WARNING: yellow + fmt + reset,
            logging.ERROR: red + fmt + reset,
            logging.CRITICAL: bold_red + fmt + reset,
        }
        log_fmt = self.FORMATS.get(record.levelno)
        return logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S").format(record)


handlers = logging.getLogger().handlers

for h in handlers:
    if isinstance(h, logging.StreamHandler):
        handler_console = h
        break
if handler_console is None:
    handler_console = logging.StreamHandler()

if handler_console is not None:
    # first we need to remove to avoid duplication
    logging.getLogger().removeHandler(handler_console)
    log = logging.getLogger(__name__)
    log.setLevel(LOG_LEVEL)
    log.addHandler(CustomHandler())


def dict_log_format(data: dict) -> str:
    """Return formatted dictionary for pretty printing inside log messages."""
    return "\n" + json.dumps(data, indent=4)
