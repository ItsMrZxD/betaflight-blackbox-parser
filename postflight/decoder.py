"""Frame decoding: turning the binary body of a session into rows of numbers.

Two things make this more than a struct-unpack loop.

**Prediction.** Almost no field is stored as an absolute value. The firmware
subtracts a prediction (the previous value, a straight-line extrapolation of
the last two, motor 0, the battery reference, ...) and stores only the
residual, which is usually tiny and so encodes in one or two bytes. Decoding
means re-adding the prediction with exactly the arithmetic the firmware used,
including 32-bit wraparound.

**Corruption.** Logs come off dataflash or an SD card on a vehicle that is
frequently crashed, and the write path drops bytes when the buffer overruns.
A single bad byte desynchronises the stream, so the decoder validates each
frame and, when one looks wrong, scans forward for the next intra frame and
carries on. Losing a slice of a log is normal; refusing to read the rest of
it is not.
"""

from __future__ import annotations

import os
import struct

from .encoding import (
    GROUP_SIZE,
    NEG_14BIT,
    NULL,
    SIGNED_VB,
    TAG2_3S32,
    TAG2_3SVARIABLE,
    TAG8_4S16,
    TAG8_8SVB,
    UNSIGNED_VB,
    Reader,
    sign_extend,
)
from .errors import CorruptLog, EndOfStream, UnsupportedLog
from .header import LOG_MARKER, parse_header

PREDICT_ZERO = 0
PREDICT_PREVIOUS = 1
PREDICT_STRAIGHT_LINE = 2
PREDICT_AVERAGE_2 = 3
PREDICT_MINTHROTTLE = 4
PREDICT_MOTOR_0 = 5
PREDICT_INCREMENT = 6
PREDICT_HOME_COORD = 7
PREDICT_1500 = 8
PREDICT_VBATREF = 9
PREDICT_LAST_MAIN_FRAME_TIME = 10
PREDICT_MINMOTOR = 11

#: Sanity bounds on how far a main frame may advance from the one before it.
#: A frame claiming a larger jump is treated as corruption rather than
#: believed, which is what stops a garbage candidate being accepted during
#: resynchronisation. Both values are the reference decoder's
#: (MAXIMUM_TIME_JUMP_BETWEEN_FRAMES and MAXIMUM_ITERATION_JUMP_BETWEEN_FRAMES
#: in blackbox-tools parser.c).
#:
#: The iteration bound has to clear the largest *legitimate* gap, which is one
#: intra-frame interval times the inter-frame denominator -- 256 * 16 = 4096 on
#: the real logs tested. Anything much looser than that simply lets more
#: corrupt frames through.
MAX_TIME_JUMP_US = 10 * 1000 * 1000
MAX_ITERATION_JUMP = 500 * 10

EVENT_SYNC_BEEP = 0
EVENT_INFLIGHT_ADJUSTMENT = 13
EVENT_LOGGING_RESUME = 14
EVENT_DISARM = 15
EVENT_FLIGHT_MODE = 30
EVENT_LOG_END = 255

#: Written verbatim by the firmware after the log-end event byte.
LOG_END_MARKER = b"End of log" + bytes([0])

EVENT_NAMES = {
    EVENT_SYNC_BEEP: "syncBeep",
    EVENT_INFLIGHT_ADJUSTMENT: "inflightAdjustment",
    EVENT_LOGGING_RESUME: "loggingResume",
    EVENT_DISARM: "disarm",
    EVENT_FLIGHT_MODE: "flightMode",
    EVENT_LOG_END: "logEnd",
}

#: Betaflight's flightLogDisarmReason_e.
DISARM_REASONS = {
    0: "ARMING_DISABLED",
    1: "FAILSAFE",
    2: "THROTTLE_TIMEOUT",
    3: "STICKS",
    4: "SWITCH",
    5: "CRASH_PROTECTION",
    6: "RUNAWAY_TAKEOFF",
    7: "GPS_RESCUE",
    8: "SERIAL_IO",
}

FRAME_KINDS = frozenset("IPSGHE")


class Frame:
    """One decoded row: a frame kind plus its field values."""

    __slots__ = ("kind", "values", "_index", "offset")

    def __init__(self, kind, values, index, offset):
        self.kind = kind
        self.values = values
        self._index = index
        self.offset = offset

    def __getitem__(self, name):
        return self.values[self._index[name]]

    def __contains__(self, name):
        return name in self._index

    def get(self, name, default=None):
        position = self._index.get(name)
        return default if position is None else self.values[position]

    def as_dict(self):
        return {name: self.values[i] for name, i in self._index.items()}

    def __repr__(self):
        return "Frame(%r, %d values)" % (self.kind, len(self.values))


class Event:
    """A decoded event frame: arming, mode change, log end and friends."""

    __slots__ = ("code", "name", "data", "offset")

    def __init__(self, code, name, data, offset):
        self.code = code
        self.name = name
        self.data = data
        self.offset = offset

    kind = "E"

    def get(self, key, default=None):
        return self.data.get(key, default)

    def __repr__(self):
        return "Event(%r, %r)" % (self.name, self.data)


class DecodeStats:
    """Bookkeeping about how cleanly a session decoded."""

    def __init__(self):
        self.frame_counts = {}
        self.corrupt_frames = 0
        self.corrupt_bytes = 0
        self.resyncs = 0
        self.unknown_frame_types = {}
        self.reached_log_end = False

    def _count(self, kind):
        self.frame_counts[kind] = self.frame_counts.get(kind, 0) + 1

    @property
    def total_frames(self):
        return sum(self.frame_counts.values())

    @property
    def corruption_ratio(self):
        total = self.total_frames + self.corrupt_frames
        return self.corrupt_frames / total if total else 0.0

    def __repr__(self):
        return "DecodeStats(%r, corrupt=%d)" % (self.frame_counts, self.corrupt_frames)


class Session:
    """One arm-to-disarm logging session out of a .BBL file.

    Frames are produced lazily by :meth:`frames`; a long log is a lot of rows
    and nothing here holds them all in memory unless you ask it to.
    """

    def __init__(self, data, index=0, offset=0, path=None):
        self.data = data
        self.index = index
        self.offset = offset
        self.path = path
        self.header = parse_header(data)
        self.decode_stats = DecodeStats()

    def __repr__(self):
        return "Session(#%d, %s, %d bytes)" % (
            self.index + 1,
            self.header.firmware,
            len(self.data),
        )

    def frames(self):
        """Yield :class:`Frame` and :class:`Event` objects in log order."""
        return _Decoder(self).run()

    def main_frames(self):
        """Yield only I and P frames, i.e. the flight data proper."""
        for frame in self.frames():
            if frame.kind == "I" or frame.kind == "P":
                yield frame

    def stats(self, **kwargs):
        from .stats import compute_stats

        return compute_stats(self, **kwargs)


class _Decoder:
    """The frame loop. One instance per pass over a session."""

    def __init__(self, session):
        self.session = session
        self.header = session.header
        session.decode_stats = DecodeStats()
        self.stats = session.decode_stats
        self.reader = Reader(session.data, session.header.end_offset)

        self.previous = None
        self.previous2 = None
        self.last_main_time = 0
        self.last_main_iteration = None
        self.skipped_frames = 0
        self.gps_home = [0, 0]

        self.time_rollover = 0

        num, den = self.header.p_interval
        # A malformed rate (0/16, 5/2, negatives) would make the skipped-frame
        # count meaningless and throw the whole log away, so fall back to 1/1.
        self.p_num, self.p_den = (num, den) if 1 <= num <= den else (1, 1)

        self.main_index = self.header.frames["I"].index
        self.time_field = self.main_index.get("time")
        self.iteration_field = self.main_index.get("loopIteration")
        self.motor0_field = self.main_index.get("motor[0]")

        # TAG8_4S16 changed layout between data versions, and reading one with
        # the other's rules produces wrong numbers rather than an error, so the
        # choice is made once here from the header.
        self.read_tag8_4s16 = (
            self.reader.read_tag8_4s16_v1
            if self.header.data_version < 2
            else self.reader.read_tag8_4s16
        )
        self._reject_unsupported_encodings()

    def _reject_unsupported_encodings(self):
        """Refuse a log outright rather than guess at an unknown encoding.

        Encoding 10 (TAG2_3SVARIABLE) is declared in Betaflight's enum but is
        never emitted by the firmware and is not implemented by blackbox-tools,
        so there is no reference for its byte layout. Decoding it by guesswork
        would produce plausible, wrong flight data.
        """
        for kind, frame_def in sorted(self.header.frames.items()):
            for field in frame_def.fields:
                if field.encoding == TAG2_3SVARIABLE:
                    raise UnsupportedLog(
                        "frame %s field %r uses encoding 10 (TAG2_3SVARIABLE), "
                        "which has no reference implementation to verify "
                        "against; refusing to guess at its layout"
                        % (kind, field.name)
                    )

    # -- main loop -------------------------------------------------------
    def run(self):
        reader = self.reader
        while not reader.eof():
            start = reader.pos
            kind = chr(reader.read_byte())
            if kind not in FRAME_KINDS:
                # A byte where a frame type should be means the stream has
                # desynchronised; count it with the other damage.
                self.stats.corrupt_frames += 1
                self.stats.unknown_frame_types[kind] = (
                    self.stats.unknown_frame_types.get(kind, 0) + 1
                )
                yield from self._resync(start)
                continue
            try:
                item = self._decode_frame(kind, start)
            except (CorruptLog, EndOfStream):
                self.stats.corrupt_frames += 1
                yield from self._resync(start)
                continue

            if item is None:  # log end
                self.stats._count("E")
                self.stats.reached_log_end = True
                return
            if isinstance(item, Frame) and not self._accept(item):
                self.stats.corrupt_frames += 1
                yield from self._resync(start)
                continue

            self.stats._count(kind)
            yield item

    def _decode_frame(self, kind, start):
        if kind == "E":
            return self._decode_event(start)
        frame_def = self.header.frames.get(kind)
        if frame_def is None:
            raise CorruptLog("frame type %r is not defined by the header" % kind)

        if kind == "P":
            if self.previous is None:
                raise CorruptLog("inter frame before any intra frame")
            self.skipped_frames = self._count_skipped_frames()
            previous, previous2 = self.previous, self.previous2
        else:
            # Intra, slow, GPS and GPS-home frames are all self-contained.
            previous = previous2 = None

        values = self._decode_fields(frame_def, previous, previous2)
        frame = Frame(kind, values, frame_def.index, start + self.session.offset)

        if kind == "H":
            self.gps_home = [
                frame.get("GPS_home[0]", 0),
                frame.get("GPS_home[1]", 0),
            ]
        return frame

    def _decode_fields(self, frame_def, previous, previous2):
        """Read one frame's worth of fields, applying predictors as we go."""
        fields = frame_def.fields
        count = len(fields)
        values = [0] * count
        reader = self.reader
        i = 0

        while i < count:
            encoding = fields[i].encoding
            if encoding == NULL:
                group = [0]
            elif encoding == SIGNED_VB:
                group = [reader.read_signed_vb()]
            elif encoding == UNSIGNED_VB:
                group = [reader.read_unsigned_vb()]
            elif encoding == NEG_14BIT:
                group = [reader.read_neg_14bit()]
            elif encoding == TAG8_8SVB:
                # This group runs until the encoding changes, up to 8 fields.
                span = 1
                while span < 8 and i + span < count:
                    if fields[i + span].encoding != TAG8_8SVB:
                        break
                    span += 1
                group = reader.read_tag8_8svb(span)
            elif encoding == TAG2_3S32:
                group = reader.read_tag2_3s32()
            elif encoding == TAG8_4S16:
                group = self.read_tag8_4s16()
            else:
                raise CorruptLog(
                    "field %r uses unknown encoding %d" % (fields[i].name, encoding)
                )

            expected = GROUP_SIZE.get(encoding, 1)
            if encoding == TAG8_8SVB:
                expected = len(group)
            if i + expected > count:
                raise CorruptLog(
                    "encoding %d at field %d overruns the %d-field frame"
                    % (encoding, i, count)
                )

            for offset, raw in enumerate(group):
                position = i + offset
                field = fields[position]
                values[position] = self._predict(
                    field, position, raw, values, previous, previous2
                )
            i += expected
        return values

    def _predict(self, field, position, value, current, previous, previous2):
        predictor = field.predictor

        if predictor == PREDICT_PREVIOUS:
            if previous is not None:
                value += previous[position]
        elif predictor == PREDICT_STRAIGHT_LINE:
            if previous is not None:
                value += 2 * previous[position] - previous2[position]
        elif predictor == PREDICT_AVERAGE_2:
            if previous is not None:
                # C integer division truncates toward zero; Python floors.
                total = previous[position] + previous2[position]
                value += -((-total) // 2) if total < 0 else total // 2
        elif predictor == PREDICT_INCREMENT:
            value += self.skipped_frames + 1
            if previous is not None:
                value += previous[position]
        elif predictor == PREDICT_MINTHROTTLE:
            value += self.header.min_throttle
        elif predictor == PREDICT_1500:
            value += 1500
        elif predictor == PREDICT_VBATREF:
            value += self.header.vbat_ref
        elif predictor == PREDICT_MOTOR_0:
            if self.motor0_field is not None:
                value += current[self.motor0_field]
        elif predictor == PREDICT_MINMOTOR:
            low = self.header.motor_output_range[0]
            value += low if low is not None else self.header.min_throttle
        elif predictor == PREDICT_HOME_COORD:
            axis = _coord_axis(field.name)
            if axis is not None and axis < len(self.gps_home):
                value += self.gps_home[axis]
        elif predictor == PREDICT_LAST_MAIN_FRAME_TIME:
            value += self.last_main_time
        # PREDICT_ZERO and anything unrecognised leave the value alone.

        value &= 0xFFFFFFFF  # the firmware computed this in 32-bit registers
        return sign_extend(value, 32) if field.signed else value

    # -- validation and recovery ----------------------------------------
    def _accept(self, frame):
        """Sanity-check a main frame and roll the history forward."""
        if frame.kind not in ("I", "P"):
            return True

        values = frame.values
        iteration = (
            values[self.iteration_field] if self.iteration_field is not None else 0
        )
        if self.time_field is None:
            time, rollover = 0, self.time_rollover
        else:
            time, rollover = self._unwrap_time(values[self.time_field])

        if self.last_main_iteration is not None:
            if not 0 <= time - self.last_main_time < MAX_TIME_JUMP_US:
                return False
            if not 0 <= iteration - self.last_main_iteration < MAX_ITERATION_JUMP:
                return False

        # Only commit the rollover once the frame has been accepted, so a
        # rejected candidate during resynchronisation cannot skew the clock.
        self.time_rollover = rollover
        if self.time_field is not None:
            values[self.time_field] = time
        self.last_main_time = time
        self.last_main_iteration = iteration
        if frame.kind == "I":
            # An intra frame is absolute, so it becomes the whole history.
            self.previous = self.previous2 = values
        else:
            self.previous2 = self.previous
            self.previous = values
        return True

    def _unwrap_time(self, raw):
        """Recover a monotonic timestamp from the 32-bit microsecond counter.

        ``micros()`` on the flight controller wraps every 71.6 minutes, and a
        log written across that boundary would otherwise look like it jumped
        backwards by an hour, taking every later frame down with it. Follows
        the reference decoder: a wrap is a raw value that moved backwards but
        by only a plausible amount in modulo-32-bit arithmetic.

        Returns ``(unwrapped_time, rollover)`` without committing the rollover,
        so a rejected frame leaves the clock untouched.
        """
        rollover = self.time_rollover
        if self.last_main_iteration is not None:
            previous_raw = self.last_main_time & 0xFFFFFFFF
            stepped_back = raw < previous_raw
            small_step = ((raw - previous_raw) & 0xFFFFFFFF) < MAX_TIME_JUMP_US
            if stepped_back and small_step:
                rollover += 1 << 32
        return raw + rollover, rollover

    def _count_skipped_frames(self):
        """How many inter frames the firmware intentionally left out.

        Betaflight logs only `p_num` out of every `p_den` loop iterations at
        low logging rates; the increment predictor has to know about the gap.
        """
        if self.last_main_iteration is None:
            return 0
        count = 0
        iteration = self.last_main_iteration + 1
        while (iteration % self.p_den) >= self.p_num:
            count += 1
            iteration += 1
            if count > MAX_ITERATION_JUMP:  # pathological header, give up
                break
        return count

    def _resync(self, frame_start):
        """Scan forward for the next intra frame and resume decoding there.

        Yields nothing itself -- it is a generator only so callers can write
        ``yield from`` uniformly -- but leaves the reader on a frame that
        decoded cleanly.
        """
        data = self.session.data
        search = frame_start + 1
        while True:
            found = data.find(b"I", search)
            if found == -1:
                self.stats.corrupt_bytes += len(data) - frame_start
                self.reader.pos = len(data)
                return
            self.reader.pos = found + 1
            try:
                frame = self._decode_frame("I", found)
            except (CorruptLog, EndOfStream):
                search = found + 1
                continue
            if frame is None or not self._accept(frame):
                search = found + 1
                continue

            self.stats.corrupt_bytes += found - frame_start
            self.stats.resyncs += 1
            self.stats._count("I")
            yield frame
            return

    # -- events ----------------------------------------------------------
    def _decode_event(self, start):
        reader = self.reader
        code = reader.read_byte()
        offset = start + self.session.offset
        name = EVENT_NAMES.get(code, "unknown(%d)" % code)

        if code == EVENT_SYNC_BEEP:
            return Event(code, name, {"time": reader.read_unsigned_vb()}, offset)
        if code == EVENT_LOGGING_RESUME:
            return Event(
                code,
                name,
                {
                    "iteration": reader.read_unsigned_vb(),
                    "time": reader.read_unsigned_vb(),
                },
                offset,
            )
        if code == EVENT_DISARM:
            reason = reader.read_unsigned_vb()
            return Event(
                code,
                name,
                {
                    "reason": reason,
                    "reasonName": DISARM_REASONS.get(reason, "UNKNOWN(%d)" % reason),
                    "time": self.last_main_time,
                },
                offset,
            )
        if code == EVENT_FLIGHT_MODE:
            new_flags = reader.read_unsigned_vb()
            last_flags = reader.read_unsigned_vb()
            return Event(
                code,
                name,
                {
                    "flags": new_flags,
                    "lastFlags": last_flags,
                    "time": self.last_main_time,
                },
                offset,
            )
        if code == EVENT_INFLIGHT_ADJUSTMENT:
            function = reader.read_byte()
            if function & 0x80:
                raw = bytes(reader.read_byte() for _ in range(4))
                value = struct.unpack("<f", raw)[0]
            else:
                value = reader.read_signed_vb()
            return Event(
                code,
                name,
                {"function": function & 0x7F, "value": value, "time": self.last_main_time},
                offset,
            )
        if code == EVENT_LOG_END:
            # The firmware writes the literal "End of log" NUL after the event
            # byte. Without checking it, any stray 0x45 0xFF pair in corrupted
            # data ends the session early while reporting a clean finish.
            marker = reader.data[reader.pos : reader.pos + len(LOG_END_MARKER)]
            if marker != LOG_END_MARKER:
                raise CorruptLog(
                    "end-of-log marker at offset %d is malformed (%r); treating "
                    "it as corruption" % (start, bytes(marker[:11]))
                )
            reader.pos += len(LOG_END_MARKER)
            return None
        raise CorruptLog("unknown event type %d at offset %d" % (code, start))


def _coord_axis(name):
    """Extract N from a field named like ``GPS_coord[N]``."""
    if name.endswith("]") and "[" in name:
        try:
            return int(name[name.rindex("[") + 1 : -1])
        except ValueError:
            return None
    return None


def split_sessions(data):
    """Split a .BBL file into the byte range of each logging session.

    Returns a list of ``(start, end)`` offsets. Flight controllers append
    every arm-to-disarm cycle to the same file, each one starting with a
    fresh header block.
    """
    starts = []
    position = data.find(LOG_MARKER)
    while position != -1:
        starts.append(position)
        position = data.find(LOG_MARKER, position + 1)
    if not starts:
        return []
    bounds = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(data)
        bounds.append((start, end))
    return bounds


def decode_session(data, index=0, offset=0, path=None):
    """Build a :class:`Session` from the bytes of a single session."""
    return Session(data, index=index, offset=offset, path=path)


def read_log(path, skip_broken=True):
    """Open a .BBL/.BFL file and return one :class:`Session` per flight.

    A session whose header will not parse is skipped rather than fatal --
    the usable flights in the same file are the point.
    """
    with open(path, "rb") as handle:
        data = handle.read()

    sessions = []
    for index, (start, end) in enumerate(split_sessions(data)):
        try:
            sessions.append(
                decode_session(data[start:end], index=index, offset=start, path=path)
            )
        except CorruptLog:
            if not skip_broken:
                raise
    if not sessions and not skip_broken:
        raise CorruptLog("%s contains no blackbox sessions" % os.path.basename(path))
    return sessions
