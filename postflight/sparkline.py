"""Tiny terminal charts, so a flight can be read without leaving the shell.

Unicode block glyphs look best, but a Windows console running a legacy code
page raises on them, so every drawing function degrades to ASCII when the
output stream cannot carry the blocks.
"""

from __future__ import annotations

import sys

BLOCKS = "▁▂▃▄▅▆▇█"
ASCII_RAMP = "._-~=+*#"
BAR_FULL = "█"
BAR_EMPTY = "░"


def supports_blocks(stream=None):
    """Can this stream actually print the block glyphs?"""
    stream = stream if stream is not None else sys.stdout
    encoding = getattr(stream, "encoding", None)
    if not encoding:
        return False
    try:
        (BLOCKS + BAR_EMPTY).encode(encoding)
    except (UnicodeEncodeError, LookupError):
        return False
    return True


def resample(values, width):
    """Bucket `values` down to `width` points by averaging each bucket."""
    if not values:
        return []
    if len(values) <= width:
        return list(values)
    out = []
    for i in range(width):
        start = i * len(values) // width
        end = max(start + 1, (i + 1) * len(values) // width)
        chunk = values[start:end]
        out.append(sum(chunk) / len(chunk))
    return out


def sparkline(values, width=60, unicode=None, low=None, high=None):
    """Render `values` as a one-line chart."""
    points = resample(values, width)
    if not points:
        return ""
    ramp = BLOCKS if (supports_blocks() if unicode is None else unicode) else ASCII_RAMP
    bottom = min(points) if low is None else low
    top = max(points) if high is None else high
    span = top - bottom
    if span <= 0:
        return ramp[0] * len(points)
    scale = len(ramp) - 1
    return "".join(
        ramp[int(round((value - bottom) / span * scale))] for value in points
    )


def bar(fraction, width=20, unicode=None):
    """A simple proportional bar for a 0..1 value."""
    fraction = max(0.0, min(1.0, fraction))
    blocks = supports_blocks() if unicode is None else unicode
    full, empty = (BAR_FULL, BAR_EMPTY) if blocks else ("#", ".")
    filled = int(round(fraction * width))
    return full * filled + empty * (width - filled)


def chart(label, values, width=60, unit="", unicode=None, low=None, high=None):
    """A labelled sparkline with its own min/max annotation."""
    if not values:
        return "%-10s (no data)" % label
    line = sparkline(values, width, unicode, low, high)
    return "%-10s %s  %s%s..%s%s" % (
        label,
        line,
        _fmt(min(values)),
        unit,
        _fmt(max(values)),
        unit,
    )


def _fmt(value):
    if abs(value) >= 100:
        return "%.0f" % value
    if abs(value) >= 10:
        return "%.1f" % value
    return "%.2f" % value
