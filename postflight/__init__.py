"""Decoder and analysis toolkit for Betaflight blackbox flight logs.

Public entry points::

    from postflight import read_log, split_sessions

    for session in read_log("LOG00001.BBL"):
        print(session.header.craft_name, session.stats().summary())
"""

from .decoder import Session, decode_session, read_log, split_sessions
from .errors import BlackboxError, CorruptLog, EndOfStream, UnsupportedLog
from .header import Header

__version__ = "1.0.0"

__all__ = [
    "BlackboxError",
    "CorruptLog",
    "EndOfStream",
    "Header",
    "Session",
    "UnsupportedLog",
    "__version__",
    "decode_session",
    "read_log",
    "split_sessions",
]
