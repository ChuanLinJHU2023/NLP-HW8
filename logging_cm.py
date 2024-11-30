"""Provides a context manager for locally changing logging levels."""

# Copied from https://docs.python.org/3/howto/logging-cookbook.html

import logging
import sys

class LoggingContext:
    def __init__(self, logger, level=None, handler=None, close=True):
        
        # Made the constructor a little easier to use: `logger` and `level` args can now be strings.

        if isinstance(logger, str):
            # look up the logger by that name
            logger = logging.getLogger(logger)   
        self.logger = logger
        
        if level=="DEBUG": level = logging.DEBUG
        elif level=="INFO": level = logging.INFO
        elif level=="WARNING": level = logging.WARNING
        self.level = level
        
        self.handler = handler
        self.close = close

    def __enter__(self):
        if self.level is not None:
            self.old_level = self.logger.level
            self.logger.setLevel(self.level)
        if self.handler:
            self.logger.addHandler(self.handler)

    def __exit__(self, et, ev, tb):
        if self.level is not None:
            self.logger.setLevel(self.old_level)
        if self.handler:
            self.logger.removeHandler(self.handler)
        if self.handler and self.close:
            self.handler.close()
        # implicit return of None => don't swallow exceptions
