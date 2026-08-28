"""Exception types raised while reading a blackbox log."""


class BlackboxError(Exception):
    """Base class for every error raised by this package."""


class EndOfStream(BlackboxError):
    """The decoder ran off the end of the byte stream mid-value."""


class CorruptLog(BlackboxError):
    """The bytes read did not make sense for the format.

    Recoverable: the frame decoder catches this, discards the partial frame
    and resynchronises on the next intra frame.
    """


class UnsupportedLog(BlackboxError):
    """The log is well formed but uses something this decoder cannot handle."""
