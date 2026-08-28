"""Generate blackbox logs that look like something a quad actually flew.

Nobody can test a log decoder without logs, and a repository is a bad place
to keep a 40 MB binary off an SD card. So this module writes them: it models
a short flight, then encodes it with the same field definitions, predictors
and encodings that Betaflight 4.x uses, producing a byte-for-byte plausible
.BBL file.

Two things use it. The test suite compares decoded output against the flight
it asked for, which exercises every encoding and predictor in the decoder.
And ``postflight demo`` writes one out so the tool can be tried immediately
after cloning, with no flight controller anywhere near it.

Caveat worth stating plainly: a log this module wrote proves the decoder is
self-consistent, not that it agrees with Betaflight. The golden-byte tests in
``tests/test_encoding.py`` and the hand-built fixture in
``tests/test_decoder.py`` are what pin the format down against the spec.
"""

from __future__ import annotations

import math
import random

from .decoder import (
    EVENT_DISARM,
    EVENT_FLIGHT_MODE,
    EVENT_LOG_END,
    EVENT_SYNC_BEEP,
    PREDICT_AVERAGE_2,
    PREDICT_HOME_COORD,
    PREDICT_INCREMENT,
    PREDICT_LAST_MAIN_FRAME_TIME,
    PREDICT_MINMOTOR,
    PREDICT_MOTOR_0,
    PREDICT_PREVIOUS,
    PREDICT_STRAIGHT_LINE,
    PREDICT_VBATREF,
    PREDICT_ZERO,
)
from .encoding import (
    NEG_14BIT,
    NULL,
    SIGNED_VB,
    TAG2_3S32,
    TAG8_4S16,
    TAG8_8SVB,
    UNSIGNED_VB,
    Writer,
)


class Field:
    """A column definition, in the same shape Betaflight declares them."""

    __slots__ = ("name", "signed", "i_pred", "i_enc", "p_pred", "p_enc")

    def __init__(self, name, signed, i_pred, i_enc, p_pred=None, p_enc=None):
        self.name = name
        self.signed = signed
        self.i_pred = i_pred
        self.i_enc = i_enc
        self.p_pred = p_pred
        self.p_enc = p_enc


def _clamp_int16(value):
    return max(-32767, min(32767, int(value)))


def _axes(name, count, signed, i_pred, i_enc, p_pred, p_enc):
    return [
        Field("%s[%d]" % (name, i), signed, i_pred, i_enc, p_pred, p_enc)
        for i in range(count)
    ]


#: Main (I/P) frame layout, matching Betaflight 4.x with debug logging off.
MAIN_FIELDS = (
    [
        Field("loopIteration", False, PREDICT_ZERO, UNSIGNED_VB, PREDICT_INCREMENT, NULL),
        Field("time", False, PREDICT_ZERO, UNSIGNED_VB, PREDICT_STRAIGHT_LINE, SIGNED_VB),
    ]
    + _axes("axisP", 3, True, PREDICT_ZERO, SIGNED_VB, PREDICT_PREVIOUS, SIGNED_VB)
    + _axes("axisI", 3, True, PREDICT_ZERO, SIGNED_VB, PREDICT_PREVIOUS, TAG2_3S32)
    + _axes("axisD", 2, True, PREDICT_ZERO, SIGNED_VB, PREDICT_PREVIOUS, SIGNED_VB)
    + _axes("rcCommand", 4, True, PREDICT_ZERO, SIGNED_VB, PREDICT_PREVIOUS, TAG8_4S16)
    + [
        Field("vbatLatest", False, PREDICT_VBATREF, NEG_14BIT, PREDICT_PREVIOUS, TAG8_8SVB),
        Field("amperageLatest", True, PREDICT_ZERO, SIGNED_VB, PREDICT_PREVIOUS, TAG8_8SVB),
        Field("rssi", False, PREDICT_ZERO, UNSIGNED_VB, PREDICT_PREVIOUS, TAG8_8SVB),
    ]
    + _axes("gyroADC", 3, True, PREDICT_ZERO, SIGNED_VB, PREDICT_AVERAGE_2, SIGNED_VB)
    + _axes("accSmooth", 3, True, PREDICT_ZERO, SIGNED_VB, PREDICT_AVERAGE_2, SIGNED_VB)
    + [
        # Motor 0 is stored against the idle value, the rest against motor 0,
        # because the four of them track each other closely.
        Field("motor[0]", False, PREDICT_MINMOTOR, UNSIGNED_VB,
              PREDICT_AVERAGE_2, SIGNED_VB),
        Field("motor[1]", False, PREDICT_MOTOR_0, SIGNED_VB,
              PREDICT_AVERAGE_2, SIGNED_VB),
        Field("motor[2]", False, PREDICT_MOTOR_0, SIGNED_VB,
              PREDICT_AVERAGE_2, SIGNED_VB),
        Field("motor[3]", False, PREDICT_MOTOR_0, SIGNED_VB,
              PREDICT_AVERAGE_2, SIGNED_VB),
    ]
)

SLOW_FIELDS = [
    Field("flightModeFlags", False, PREDICT_ZERO, UNSIGNED_VB),
    Field("stateFlags", False, PREDICT_ZERO, UNSIGNED_VB),
    Field("failsafePhase", False, PREDICT_ZERO, TAG2_3S32),
    Field("rxSignalReceived", False, PREDICT_ZERO, TAG2_3S32),
    Field("rxFlightChannelsValid", False, PREDICT_ZERO, TAG2_3S32),
]

GPS_FIELDS = [
    Field("time", False, PREDICT_LAST_MAIN_FRAME_TIME, UNSIGNED_VB),
    Field("GPS_numSat", False, PREDICT_ZERO, UNSIGNED_VB),
    Field("GPS_coord[0]", True, PREDICT_HOME_COORD, SIGNED_VB),
    Field("GPS_coord[1]", True, PREDICT_HOME_COORD, SIGNED_VB),
    Field("GPS_altitude", False, PREDICT_ZERO, UNSIGNED_VB),
    Field("GPS_speed", False, PREDICT_ZERO, UNSIGNED_VB),
    Field("GPS_ground_course", False, PREDICT_ZERO, UNSIGNED_VB),
]

GPS_HOME_FIELDS = [
    Field("GPS_home[0]", True, PREDICT_ZERO, SIGNED_VB),
    Field("GPS_home[1]", True, PREDICT_ZERO, SIGNED_VB),
]

VBAT_REF = 1680  # 4S pack straight off the charger, in centivolts
GYRO_SCALE_DPS = 1 / 16.4  # degrees per second per raw count, a 2000 dps gyro
ACC_1G = 2048  # raw counts per g, matching the acc_1G header
MOTOR_MIN = 0
MOTOR_MAX = 2047
ARMED_MODE_FLAGS = 0b1  # bit 0 is ARM
ANGLE_MODE_FLAGS = 0b11  # ARM + ANGLE


class FlightModel:
    """A crude but plausible freestyle flight, sampled per loop iteration."""

    def __init__(self, duration=12.0, loop_hz=1000, seed=7, crash_at=None,
                 rx_glitch_at=None):
        self.duration = duration
        self.loop_hz = loop_hz
        self.rng = random.Random(seed)
        self.crash_at = crash_at
        self.rx_glitch_at = rx_glitch_at
        self.iterations = int(duration * loop_hz)

    def throttle(self, t):
        """Normalised 0..1 stick position over the course of the flight."""
        if t < 0.8:
            return 0.05  # armed, sat on the ground
        if self.crash_at is not None and t > self.crash_at:
            return 0.0  # pilot chopped the throttle
        base = 0.45 + 0.18 * math.sin(t * 1.7)
        punch = 0.30 * max(0.0, math.sin(t * 0.9 - 1.0)) ** 3
        return min(1.0, base + punch)

    def sample(self, iteration):
        rng = self.rng
        t = iteration / self.loop_hz
        throttle = self.throttle(t)
        crashing = self.crash_at is not None and self.crash_at <= t < self.crash_at + 0.4

        # Model the sensors in physical units, then convert to raw counts the
        # way the gyro and accelerometer drivers would.
        spin_dps = 40 + 90 * throttle  # ordinary freestyle rotation rates
        gyro_dps = [rng.gauss(0, spin_dps) for _ in range(3)]
        acc_g = [rng.gauss(0, 0.35), rng.gauss(0, 0.35),
                 0.9 + 1.6 * throttle + rng.gauss(0, 0.2)]
        if crashing:
            # tumbling hard: rate-limited on one axis, several g through the frame
            gyro_dps = [rng.gauss(0, 700) for _ in range(3)]
            gyro_dps[0] = math.copysign(1500 + rng.random() * 400, gyro_dps[0] or 1.0)
            acc_g = [rng.gauss(0, 4.0), rng.gauss(0, 4.0), rng.gauss(0, 5.0)]
            acc_g[2] = math.copysign(max(7.0, abs(acc_g[2])), acc_g[2] or 1.0)

        # Both sensors are 16-bit; a hard enough crash rails them, which is
        # exactly the clipping the analyser looks for.
        gyro = [_clamp_int16(value / GYRO_SCALE_DPS) for value in gyro_dps]
        acc = [_clamp_int16(value * ACC_1G) for value in acc_g]

        # Battery: static drain plus sag proportional to current draw.
        amperage = int((80 + 4800 * throttle**1.5) * (0.9 + 0.2 * rng.random()))
        sag = amperage * 0.022
        vbat = int(VBAT_REF - 9.5 * t - sag)

        rssi = 1023
        glitch = self.rx_glitch_at
        if glitch is not None and glitch <= t < glitch + 0.5:
            rssi = int(120 + rng.random() * 60)

        stick = [int(500 * math.sin(t * 2.1)), int(400 * math.cos(t * 1.3)),
                 int(300 * math.sin(t * 0.7))]
        rc = [stick[0], stick[1], stick[2], 1000 + int(1000 * throttle)]

        mix = [
            throttle + 0.05 * math.sin(t * 3.0),
            throttle - 0.05 * math.sin(t * 3.0),
            throttle + 0.04 * math.cos(t * 2.2),
            throttle - 0.04 * math.cos(t * 2.2),
        ]
        if crashing:  # one motor pinned wide open while the FC fights the tumble
            mix[0] = 1.0
        motors = [
            max(MOTOR_MIN, min(MOTOR_MAX, int(MOTOR_MAX * value))) for value in mix
        ]

        pid_p = [int(gyro_dps[i] * -6.0) for i in range(3)]
        pid_i = [int(math.sin(t * 0.5 + i) * 90) for i in range(3)]
        pid_d = [int(rng.gauss(0, 45)) for i in range(2)]

        return {
            "loopIteration": iteration,
            "time": int(iteration * (1_000_000 / self.loop_hz)),
            "axisP[0]": pid_p[0], "axisP[1]": pid_p[1], "axisP[2]": pid_p[2],
            "axisI[0]": pid_i[0], "axisI[1]": pid_i[1], "axisI[2]": pid_i[2],
            "axisD[0]": pid_d[0], "axisD[1]": pid_d[1],
            "rcCommand[0]": rc[0], "rcCommand[1]": rc[1],
            "rcCommand[2]": rc[2], "rcCommand[3]": rc[3],
            "vbatLatest": vbat,
            "amperageLatest": amperage,
            "rssi": rssi,
            "gyroADC[0]": gyro[0], "gyroADC[1]": gyro[1], "gyroADC[2]": gyro[2],
            "accSmooth[0]": acc[0], "accSmooth[1]": acc[1], "accSmooth[2]": acc[2],
            "motor[0]": motors[0], "motor[1]": motors[1],
            "motor[2]": motors[2], "motor[3]": motors[3],
        }


class LogBuilder:
    """Encodes rows of field values into a well-formed blackbox session."""

    def __init__(self, craft_name="SynthQuad", firmware="Betaflight 4.4.0 (release) SYNTH",
                 board="BTFL SYNTHF4", loop_hz=1000, i_interval=32, gps=False):
        self.writer = Writer()
        self.craft_name = craft_name
        self.firmware = firmware
        self.board = board
        self.loop_hz = loop_hz
        self.i_interval = i_interval
        self.gps = gps

        self.previous = None
        self.previous2 = None
        self.last_main_time = 0
        self.gps_home = [0, 0]

    # -- header ----------------------------------------------------------
    def write_header(self):
        write = self._header_line
        write("Product", "Blackbox flight data recorder by Nicholas Sherlock")
        write("Data version", "2")
        write("I interval", str(self.i_interval))
        write("P interval", "1/1")
        self._field_headers("I", MAIN_FIELDS, names=True, predictor="i", encoding="i")
        self._field_headers("P", MAIN_FIELDS, names=False, predictor="p", encoding="p")
        self._field_headers("S", SLOW_FIELDS, names=True, predictor="i", encoding="i")
        if self.gps:
            self._field_headers(
                "H", GPS_HOME_FIELDS, names=True, predictor="i", encoding="i"
            )
            self._field_headers("G", GPS_FIELDS, names=True, predictor="i", encoding="i")
        write("Firmware type", "Cleanflight")
        write("Firmware revision", self.firmware)
        write("Firmware date", "Feb 24 2023 03:20:07")
        write("Board information", self.board)
        write("Craft name", self.craft_name)
        write("looptime", str(int(1_000_000 / self.loop_hz)))
        write("gyro_sync_denom", "1")
        write("pid_process_denom", "1")
        write("minthrottle", "1070")
        write("maxthrottle", "2000")
        write("gyro.scale", "0x3d79c190")  # 1/16.4 dps per LSB, i.e. a 2000 dps gyro
        write("motorOutput", "%d,%d" % (MOTOR_MIN, MOTOR_MAX))
        write("acc_1G", "2048")
        write("vbat_scale", "110")
        write("vbatcellvoltage", "330,350,430")
        write("vbatref", str(VBAT_REF))
        write("currentSensor", "0,400")
        write("rcRate", "100")
        write("Log start datetime", "2026-08-28T18:41:07.000+00:00")

    def _header_line(self, key, value):
        self.writer.write_raw(("H %s:%s\n" % (key, value)).encode("latin-1"))

    def _field_headers(self, kind, fields, names, predictor, encoding):
        if names:
            self._header_line(
                "Field %s name" % kind, ",".join(field.name for field in fields)
            )
            self._header_line(
                "Field %s signed" % kind,
                ",".join("1" if field.signed else "0" for field in fields),
            )
        self._header_line(
            "Field %s predictor" % kind,
            ",".join(
                str(field.i_pred if predictor == "i" else field.p_pred)
                for field in fields
            ),
        )
        self._header_line(
            "Field %s encoding" % kind,
            ",".join(
                str(field.i_enc if encoding == "i" else field.p_enc) for field in fields
            ),
        )

    # -- frames ----------------------------------------------------------
    def write_main_frame(self, row, intra):
        values = [row[field.name] for field in MAIN_FIELDS]
        kind = "I" if intra else "P"
        self.writer.write_byte(ord(kind))
        self._encode_fields(
            MAIN_FIELDS,
            values,
            intra=intra,
            previous=None if intra else self.previous,
            previous2=None if intra else self.previous2,
        )
        if intra:
            self.previous = self.previous2 = values
        else:
            self.previous2 = self.previous
            self.previous = values
        self.last_main_time = row["time"]

    def write_slow_frame(self, flight_mode_flags, state_flags, failsafe_phase,
                         rx_signal=1, rx_channels_valid=1):
        self.writer.write_byte(ord("S"))
        self._encode_fields(
            SLOW_FIELDS,
            [flight_mode_flags, state_flags, failsafe_phase, rx_signal, rx_channels_valid],
            intra=True,
            previous=None,
            previous2=None,
        )

    def write_gps_home(self, lat, lon):
        self.gps_home = [lat, lon]
        self.writer.write_byte(ord("H"))
        self._encode_fields(GPS_HOME_FIELDS, [lat, lon], True, None, None)

    def write_gps_frame(self, time_us, num_sat, lat, lon, altitude, speed, course):
        self.writer.write_byte(ord("G"))
        self._encode_fields(
            GPS_FIELDS,
            [time_us, num_sat, lat, lon, altitude, speed, course],
            True,
            None,
            None,
        )

    def write_event(self, code, payload=()):
        self.writer.write_byte(ord("E"))
        self.writer.write_byte(code)
        for value in payload:
            self.writer.write_unsigned_vb(value)

    def write_log_end(self):
        self.writer.write_byte(ord("E"))
        self.writer.write_byte(EVENT_LOG_END)
        self.writer.write_raw(b"End of log\x00")

    # -- encoding --------------------------------------------------------
    def _encode_fields(self, fields, values, intra, previous, previous2):
        """Subtract the prediction from each value, then encode the residual."""
        residuals = [
            self._residual(fields, i, values, intra, previous, previous2)
            for i in range(len(fields))
        ]

        writer = self.writer
        i = 0
        while i < len(fields):
            encoding = fields[i].i_enc if intra else fields[i].p_enc
            if encoding == NULL:
                i += 1
            elif encoding == SIGNED_VB:
                writer.write_signed_vb(residuals[i])
                i += 1
            elif encoding == UNSIGNED_VB:
                writer.write_unsigned_vb(residuals[i])
                i += 1
            elif encoding == NEG_14BIT:
                writer.write_neg_14bit(residuals[i])
                i += 1
            elif encoding == TAG8_8SVB:
                span = 1
                while span < 8 and i + span < len(fields):
                    other = fields[i + span]
                    if (other.i_enc if intra else other.p_enc) != TAG8_8SVB:
                        break
                    span += 1
                writer.write_tag8_8svb(residuals[i : i + span])
                i += span
            elif encoding in (TAG2_3S32,):
                writer.write_tag2_3s32(residuals[i : i + 3])
                i += 3
            elif encoding == TAG8_4S16:
                writer.write_tag8_4s16(residuals[i : i + 4])
                i += 4
            else:
                raise ValueError("cannot encode field %r" % fields[i].name)

    def _residual(self, fields, position, values, intra, previous, previous2):
        field = fields[position]
        predictor = field.i_pred if intra else field.p_pred
        value = values[position]

        if predictor == PREDICT_PREVIOUS:
            value -= previous[position] if previous else 0
        elif predictor == PREDICT_STRAIGHT_LINE:
            if previous:
                value -= 2 * previous[position] - previous2[position]
        elif predictor == PREDICT_AVERAGE_2:
            if previous:
                total = previous[position] + previous2[position]
                value -= -((-total) // 2) if total < 0 else total // 2
        elif predictor == PREDICT_INCREMENT:
            value -= (previous[position] if previous else 0) + 1
        elif predictor == PREDICT_VBATREF:
            value -= VBAT_REF
        elif predictor == PREDICT_MINMOTOR:
            value -= MOTOR_MIN
        elif predictor == PREDICT_MOTOR_0:
            value -= values[_index_of(fields, "motor[0]")]
        elif predictor == PREDICT_HOME_COORD:
            axis = 0 if field.name.endswith("[0]") else 1
            value -= self.gps_home[axis]
        elif predictor == PREDICT_LAST_MAIN_FRAME_TIME:
            value -= self.last_main_time
        return value

    def bytes(self):
        return self.writer.bytes()


def _index_of(fields, name):
    for i, field in enumerate(fields):
        if field.name == name:
            return i
    raise KeyError(name)


def build_log(duration=12.0, loop_hz=1000, seed=7, crash_at=None, rx_glitch_at=None,
              gps=False, craft_name="SynthQuad", i_interval=32, slow_interval=256,
              corrupt=0, end_cleanly=True):
    """Build one complete session and return it as bytes.

    `corrupt` overwrites that many randomly chosen bytes of the frame body
    with noise, which is how the resynchronisation path gets tested.
    """
    model = FlightModel(duration, loop_hz, seed, crash_at, rx_glitch_at)
    builder = LogBuilder(craft_name=craft_name, loop_hz=loop_hz,
                         i_interval=i_interval, gps=gps)
    builder.write_header()
    body_start = len(builder.writer)

    builder.write_event(EVENT_SYNC_BEEP, [0])
    builder.write_event(EVENT_FLIGHT_MODE, [ANGLE_MODE_FLAGS, 0])
    if gps:
        builder.write_gps_home(514_000_000, -1_000_000)

    glitch_start = rx_glitch_at
    for iteration in range(model.iterations):
        row = model.sample(iteration)
        builder.write_main_frame(row, intra=(iteration % i_interval == 0))

        if iteration % slow_interval == 0:
            t = iteration / loop_hz
            in_glitch = (
                glitch_start is not None and glitch_start <= t < glitch_start + 0.5
            )
            builder.write_slow_frame(
                ANGLE_MODE_FLAGS,
                0b1000,  # SMALL_ANGLE
                1 if in_glitch else 0,  # failsafe phase RX_LOSS_DETECTED
                0 if in_glitch else 1,
                0 if in_glitch else 1,
            )
        if gps and iteration % (loop_hz // 5) == 0:  # 5 Hz GPS
            t = iteration / loop_hz
            builder.write_gps_frame(
                row["time"],
                11,
                514_000_000 + int(t * 900),
                -1_000_000 + int(t * 600),
                int(30 + 20 * math.sin(t)),
                int(400 + 300 * math.sin(t * 0.8)),
                int((t * 40) % 360) * 10,
            )

    builder.write_event(EVENT_DISARM, [4])  # DISARM_REASON_SWITCH
    if end_cleanly:
        builder.write_log_end()

    data = bytearray(builder.bytes())
    if corrupt:
        rng = random.Random(seed + 1)
        for _ in range(corrupt):
            position = rng.randrange(body_start + 64, len(data) - 32)
            data[position] = rng.randrange(256)
    return bytes(data)


def build_multi_session_log(count=2, **kwargs):
    """Concatenate several sessions the way a flight controller does."""
    chunks = []
    for index in range(count):
        options = dict(kwargs)
        options.setdefault("craft_name", "SynthQuad")
        options["seed"] = kwargs.get("seed", 7) + index
        chunks.append(build_log(**options))
    return b"".join(chunks)
