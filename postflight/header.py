"""Parsing of the ASCII header block that opens every blackbox session.

A session begins with a run of newline-terminated ``H key:value`` lines and
then switches to binary frames. The header carries two kinds of information:

* **frame definitions** -- for each frame type, the name, signedness,
  predictor and encoding of every field. Without these the binary payload is
  undecodable, which is why a truncated header is fatal while a truncated
  frame is not.
* **system configuration** -- throttle range, battery reference, gyro scale
  and so on, needed to turn raw field values into physical units.
"""

from __future__ import annotations

import struct

from .errors import CorruptLog

#: Marks the start of a session. A single .BBL file off a flight controller
#: often holds several of these back to back, one per arm/disarm cycle.
LOG_MARKER = b"H Product:Blackbox"

_FRAME_KINDS = ("I", "P", "S", "G", "H")

#: Headers whose value is free text and must not be stripped or parsed.
_TEXT_HEADERS = {
    "Product",
    "Firmware type",
    "Firmware revision",
    "Firmware date",
    "Board information",
    "Craft name",
    "Log start datetime",
}


class FieldDef:
    """One column of one frame type."""

    __slots__ = ("name", "signed", "predictor", "encoding")

    def __init__(self, name, signed, predictor, encoding):
        self.name = name
        self.signed = signed
        self.predictor = predictor
        self.encoding = encoding

    def __repr__(self):
        return "FieldDef(%r, signed=%r, predictor=%d, encoding=%d)" % (
            self.name,
            self.signed,
            self.predictor,
            self.encoding,
        )


class FrameDef:
    """The full column layout of one frame type."""

    def __init__(self, kind, fields):
        self.kind = kind
        self.fields = fields
        self.index = {field.name: i for i, field in enumerate(fields)}

    def __len__(self):
        return len(self.fields)

    @property
    def names(self):
        return [field.name for field in self.fields]

    def __repr__(self):
        return "FrameDef(%r, %d fields)" % (self.kind, len(self.fields))


class Header:
    """Everything the header block told us about a session."""

    def __init__(self, raw, frames, end_offset):
        self.raw = raw
        self.frames = frames
        self.end_offset = end_offset

    # -- text ------------------------------------------------------------
    @property
    def craft_name(self):
        return self.raw.get("Craft name", "")

    @property
    def firmware(self):
        revision = self.raw.get("Firmware revision", "")
        return revision or self.raw.get("Firmware type", "unknown")

    @property
    def board(self):
        return self.raw.get("Board information", "")

    @property
    def firmware_date(self):
        return self.raw.get("Firmware date", "")

    @property
    def log_start_datetime(self):
        return self.raw.get("Log start datetime", "")

    @property
    def data_version(self):
        return self.int_header("Data version", 2)

    # -- numbers ---------------------------------------------------------
    @property
    def min_throttle(self):
        """Throttle value that means 'motors at idle'.

        Betaflight 4.x dropped ``minthrottle`` in favour of the digital
        ``motorOutput`` range, so fall back to the low end of that.
        """
        value = self.int_header("minthrottle")
        if value is None:
            value = self.motor_output_range[0]
        return value if value is not None else 1000

    @property
    def max_throttle(self):
        value = self.int_header("maxthrottle")
        if value is None:
            value = self.motor_output_range[1]
        return value if value is not None else 2000

    @property
    def motor_output_range(self):
        values = self.int_list("motorOutput")
        if len(values) == 2:
            return values[0], values[1]
        return None, None

    @property
    def vbat_ref(self):
        return self.int_header("vbatref", 0)

    @property
    def vbat_scale(self):
        for key in ("vbat_scale", "vbatscale"):
            value = self.int_header(key)
            if value is not None:
                return value
        return 110

    @property
    def cell_voltage_limits(self):
        """(min, warn, max) per-cell voltage in centivolts, if logged."""
        values = self.int_list("vbatcellvoltage")
        return tuple(values) if len(values) == 3 else None

    @property
    def current_sensor(self):
        """(offset, scale) of the current sensor, if one is configured."""
        values = self.int_list("currentSensor")
        return (values[0], values[1]) if len(values) == 2 else (0, 0)

    @property
    def acc_1g(self):
        return self.int_header("acc_1G", 2048)

    @property
    def gyro_scale(self):
        """Degrees per second per raw gyro unit.

        Logged as the hex bit pattern of a float, e.g. ``0x3d79c190``.
        """
        value = self.raw.get("gyro.scale") or self.raw.get("gyro_scale")
        if not value:
            return None
        try:
            bits = int(value, 16) if value.lower().startswith("0x") else int(value)
        except ValueError:
            return None
        return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]

    @property
    def looptime(self):
        return self.int_header("looptime", 0)

    @property
    def frame_interval_i(self):
        return self.int_header("I interval", 32) or 32

    @property
    def p_interval(self):
        """P frames logged per loop iteration, as a (numerator, denominator)."""
        raw = self.raw.get("P interval")
        if raw and "/" in raw:
            num, _, den = raw.partition("/")
            try:
                return int(num), int(den)
            except ValueError:
                pass
        elif raw:
            # Betaflight 4.2 writes a bare denominator here, e.g. "P interval:16"
            # meaning one inter frame per sixteen loop iterations.
            try:
                return 1, int(raw)
            except ValueError:
                pass
        denom = self.int_header("P denom")
        if denom:
            return 1, denom
        return 1, 1

    # -- helpers ---------------------------------------------------------
    def int_header(self, key, default=None):
        raw = self.raw.get(key)
        if raw is None:
            return default
        try:
            return _parse_int(raw.split(",")[0])
        except ValueError:
            return default

    def int_list(self, key):
        raw = self.raw.get(key)
        if not raw:
            return []
        values = []
        for part in raw.split(","):
            try:
                values.append(_parse_int(part))
            except ValueError:
                return []
        return values

    def has_field(self, name, kind="I"):
        frame = self.frames.get(kind)
        return bool(frame) and name in frame.index

    def __repr__(self):
        return "Header(%s, craft=%r, %d frame types)" % (
            self.firmware or "unknown",
            self.craft_name,
            len(self.frames),
        )


def _split_csv(value):
    return [part.strip() for part in value.split(",")] if value else []


def _parse_int(text):
    """Parse a header integer, tolerating zero padding.

    ``int(x, 0)`` accepts the 0x form the gyro scale uses but rejects a
    zero-padded decimal like ``010``, which would otherwise make one benign
    header line condemn the whole log.
    """
    text = text.strip()
    try:
        return int(text, 10)
    except ValueError:
        return int(text, 0)


def _ints(value, what):
    try:
        return [_parse_int(part) for part in _split_csv(value)]
    except ValueError:
        raise CorruptLog(
            "header %s is not a list of integers: %r" % (what, value)
        ) from None


def parse_header(data, start=0):
    """Read the header block of a session out of `data`.

    Returns a :class:`Header` whose ``end_offset`` is the position of the
    first binary frame.
    """
    raw = {}
    pos = start
    size = len(data)

    while pos + 1 < size and data[pos] == 0x48 and data[pos + 1] == 0x20:  # "H "
        end = data.find(b"\n", pos)
        if end == -1:
            raise CorruptLog("header line at offset %d is never terminated" % pos)
        line = data[pos + 2 : end].decode("latin-1").rstrip("\r")
        pos = end + 1
        key, sep, value = line.partition(":")
        if not sep:
            continue  # tolerate a stray line rather than abandoning the log
        raw[key.strip()] = value if key.strip() in _TEXT_HEADERS else value.strip()

    if not raw:
        raise CorruptLog("no header lines found at offset %d" % start)

    frames = _build_frames(raw)
    if "I" not in frames:
        raise CorruptLog("header defines no intra frame, log is undecodable")
    return Header(raw, frames, pos)


def _build_frames(raw):
    """Turn the ``Field X ...`` headers into :class:`FrameDef` objects."""
    frames = {}
    intra_names = _split_csv(raw.get("Field I name", ""))
    intra_signed = _ints(raw.get("Field I signed", ""), "Field I signed")

    for kind in _FRAME_KINDS:
        predictors = raw.get("Field %s predictor" % kind)
        encodings = raw.get("Field %s encoding" % kind)
        if predictors is None or encodings is None:
            continue

        # Inter (P) frames reuse the intra frame's column names and signs;
        # only their predictors and encodings differ.
        if kind == "P":
            names, signs = intra_names, intra_signed
        else:
            names = _split_csv(raw.get("Field %s name" % kind, ""))
            signs = _ints(raw.get("Field %s signed" % kind, ""), "Field %s signed" % kind)

        predictors = _ints(predictors, "Field %s predictor" % kind)
        encodings = _ints(encodings, "Field %s encoding" % kind)

        if not names or len(predictors) != len(names) or len(encodings) != len(names):
            raise CorruptLog(
                "frame %s declares %d names but %d predictors and %d encodings"
                % (kind, len(names), len(predictors), len(encodings))
            )
        if len(signs) < len(names):
            signs = signs + [0] * (len(names) - len(signs))

        frames[kind] = FrameDef(
            kind,
            [
                FieldDef(names[i], bool(signs[i]), predictors[i], encodings[i])
                for i in range(len(names))
            ],
        )
    return frames
