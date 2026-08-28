"""Flight statistics and anomaly detection over a decoded session.

Everything here is computed in one streaming pass. A ten-minute log at 8 kHz
is a few million rows, so nothing accumulates per-sample state beyond a small
decimating buffer for the charts.

The anomaly rules are deliberately conservative and each one says what it
looked at, because a tool that cries desync at every log is worse than no
tool. Thresholds live in :class:`Thresholds` so they can be argued with.
"""

from __future__ import annotations

import math

from .decoder import DISARM_REASONS, Event, Frame

#: Betaflight failsafePhase enum, logged in slow frames.
FAILSAFE_PHASES = {
    0: "IDLE",
    1: "RX_LOSS_DETECTED",
    2: "LANDING",
    3: "LANDED",
    4: "RX_LOSS_MONITORING",
    5: "RX_LOSS_RECOVERED",
}

#: Best-effort boxId order for Betaflight 4.x. The mapping is firmware
#: dependent -- treat a mode name as a hint, not gospel.
FLIGHT_MODES = [
    "ARM", "ANGLE", "HORIZON", "MAG", "HEADFREE", "PASSTHRU", "FAILSAFE",
    "GPSRESCUE", "ANTIGRAVITY", "HEADADJ", "CAMSTAB", "BEEPERON", "LEDLOW",
    "CALIB", "OSD", "TELEMETRY", "SERVO1", "SERVO2", "SERVO3", "BLACKBOX",
    "AIRMODE", "3D", "FPVANGLEMIX", "BLACKBOXERASE", "CAMERA1", "CAMERA2",
    "CAMERA3", "FLIPOVERAFTERCRASH", "PREARM", "BEEPGPSCOUNT", "VTXPITMODE",
    "PARALYZE", "USER1", "USER2", "USER3", "USER4",
]

#: Raw gyro magnitude that means the sensor railed rather than measured.
GYRO_CLIP_RAW = 32000


class Thresholds:
    """Tunable limits for the anomaly rules."""

    motor_saturation = 0.97  # fraction of the output range
    motor_saturation_ms = 50  # how long all motors must sit there
    crash_g = 6.0
    crash_gyro_dps = 900.0
    rssi_low = 0.20  # fraction of full scale
    rssi_low_ms = 100
    cell_min_v = 3.30
    dropout_pct = 1.0
    motor_imbalance = 0.12  # spread between mean motor outputs


class _Decimator:
    """Keeps at most `limit` evenly spread samples from an unbounded stream."""

    def __init__(self, limit=2048):
        self.limit = limit
        self.samples = []
        self.stride = 1
        self._seen = 0

    def add(self, value):
        if self._seen % self.stride == 0:
            self.samples.append(value)
            if len(self.samples) >= self.limit:
                self.samples = self.samples[::2]
                self.stride *= 2
        self._seen += 1

    def __len__(self):
        return len(self.samples)


class FlightStats:
    """The result of :func:`compute_stats`."""

    def __init__(self, session):
        self.session = session
        self.header = session.header
        self.thresholds = Thresholds()

        self.frame_counts = {}

        self.start_time_us = None
        self.end_time_us = None
        self.first_iteration = None
        self.last_iteration = None
        self.main_frames = 0

        self.motor_max = []
        self.motor_mean = []
        self.throttle_mean = 0.0
        self.throttle_max = 0.0
        self.saturated_samples = 0

        self.cells = 0
        self.v_start = None
        self.v_min = None
        self.v_end = None
        self.amps_max = 0.0
        self.mah_used = 0.0

        self.rssi_min = None
        self.rssi_scale = 1023.0
        self.low_rssi_samples = 0

        self.gyro_peak = [0.0, 0.0, 0.0]
        self.gyro_rms = [0.0, 0.0, 0.0]
        self.gyro_clipped = 0
        self.acc_peak_g = 0.0

        self.gps_sats = None
        self.gps_speed_max_kmh = None
        self.gps_alt_max = None
        self.gps_distance_max_m = None

        self.events = []
        self.anomalies = []
        self.series = {}

    # -- derived ---------------------------------------------------------
    @property
    def decode(self):
        """Decoder bookkeeping for the most recent pass over the session."""
        return self.session.decode_stats

    @property
    def duration_s(self):
        if self.start_time_us is None or self.end_time_us is None:
            return 0.0
        return (self.end_time_us - self.start_time_us) / 1_000_000

    @property
    def loop_hz(self):
        """Actual logging rate, from iterations covered over elapsed time."""
        if not self.duration_s or self.last_iteration is None:
            return 0.0
        span = self.last_iteration - self.first_iteration
        return span / self.duration_s if span > 0 else 0.0

    @property
    def expected_iterations(self):
        """How many frames the firmware meant to write, not how many loops it ran.

        At a logging rate of 1/16 the flight controller deliberately skips
        fifteen iterations out of every sixteen, and counting those as losses
        makes every log look like it dropped 94% of its data.
        """
        if self.first_iteration is None:
            return 0
        span = self.last_iteration - self.first_iteration + 1
        num, den = self.header.p_interval
        return max(1, span * num // den) if den > 1 else span

    @property
    def dropped_iterations(self):
        return max(0, self.expected_iterations - self.main_frames)

    @property
    def dropout_pct(self):
        expected = self.expected_iterations
        return 100.0 * self.dropped_iterations / expected if expected else 0.0

    @property
    def v_per_cell_min(self):
        return self.v_min / self.cells if self.cells and self.v_min else None

    @property
    def motor_imbalance(self):
        if len(self.motor_mean) < 2:
            return 0.0
        return max(self.motor_mean) - min(self.motor_mean)

    def as_dict(self):
        return {
            "craft": self.header.craft_name,
            "firmware": self.header.firmware,
            "board": self.header.board,
            "durationSeconds": round(self.duration_s, 3),
            "loopHz": round(self.loop_hz, 1),
            "frames": dict(self.frame_counts),
            "mainFrames": self.main_frames,
            "droppedIterations": self.dropped_iterations,
            "dropoutPercent": round(self.dropout_pct, 3),
            "corruptFrames": self.decode.corrupt_frames,
            "resyncs": self.decode.resyncs,
            "throttleMean": round(self.throttle_mean, 4),
            "throttleMax": round(self.throttle_max, 4),
            "motorMax": [round(v, 4) for v in self.motor_max],
            "motorMean": [round(v, 4) for v in self.motor_mean],
            "motorImbalance": round(self.motor_imbalance, 4),
            "cells": self.cells,
            "voltageStart": _round(self.v_start, 2),
            "voltageMin": _round(self.v_min, 2),
            "voltageEnd": _round(self.v_end, 2),
            "voltagePerCellMin": _round(self.v_per_cell_min, 3),
            "currentMaxAmps": round(self.amps_max, 2),
            "mahUsed": round(self.mah_used, 1),
            "rssiMin": self.rssi_min,
            "gyroPeakDps": [round(v, 1) for v in self.gyro_peak],
            "gyroRmsDps": [round(v, 2) for v in self.gyro_rms],
            "gyroClippedSamples": self.gyro_clipped,
            "accPeakG": round(self.acc_peak_g, 2),
            "gpsSatellites": self.gps_sats,
            "gpsSpeedMaxKmh": _round(self.gps_speed_max_kmh, 1),
            "gpsAltitudeMax": self.gps_alt_max,
            "gpsDistanceMaxMeters": _round(self.gps_distance_max_m, 1),
            "events": self.events,
            "anomalies": self.anomalies,
        }


def _round(value, digits):
    return None if value is None else round(value, digits)


def compute_stats(session, chart_points=2048, vbat_adc=False, gyro_scale=None):
    """Walk a session once and summarise the flight."""
    stats = FlightStats(session)
    header = session.header
    thresholds = stats.thresholds

    scale = gyro_scale if gyro_scale is not None else _gyro_scale(header)
    motor_low, motor_high = header.motor_output_range
    if motor_low is None:
        motor_low, motor_high = header.min_throttle, header.max_throttle
    motor_span = max(1, motor_high - motor_low)

    motor_names = _numbered_fields(header, "motor")
    gyro_names = _numbered_fields(header, "gyroADC")
    acc_names = _numbered_fields(header, "accSmooth")
    has_vbat = header.has_field("vbatLatest")
    has_amps = header.has_field("amperageLatest")
    has_rssi = header.has_field("rssi")

    motor_sum = [0.0] * len(motor_names)
    motor_max = [0.0] * len(motor_names)
    gyro_sq = [0.0] * len(gyro_names)
    gyro_peak = [0.0] * len(gyro_names)
    throttle_sum = 0.0
    amp_seconds = 0.0
    previous_time = None
    saturation_run_us = 0
    rssi_run_us = 0
    home = None

    charts = {
        "throttle": _Decimator(chart_points),
        "voltage": _Decimator(chart_points),
        "current": _Decimator(chart_points),
        "gyro": _Decimator(chart_points),
    }

    for item in session.frames():
        if isinstance(item, Event):
            _record_event(stats, item)
            continue
        if not isinstance(item, Frame):  # pragma: no cover - defensive
            continue

        kind = item.kind
        stats.frame_counts[kind] = stats.frame_counts.get(kind, 0) + 1

        if kind == "S":
            _record_slow(stats, item)
            continue
        if kind == "H":
            home = (item.get("GPS_home[0]", 0), item.get("GPS_home[1]", 0))
            continue
        if kind == "G":
            _record_gps(stats, item, home)
            continue
        if kind not in ("I", "P"):
            continue

        # -- main frame ---------------------------------------------------
        stats.main_frames += 1
        time_us = item.get("time", 0)
        iteration = item.get("loopIteration", stats.main_frames)
        if stats.start_time_us is None:
            stats.start_time_us = time_us
            stats.first_iteration = iteration
        stats.end_time_us = time_us
        stats.last_iteration = iteration

        dt_us = 0 if previous_time is None else max(0, time_us - previous_time)
        previous_time = time_us

        # motors and throttle
        if motor_names:
            saturated = True
            for i, name in enumerate(motor_names):
                fraction = (item.get(name, 0) - motor_low) / motor_span
                fraction = min(1.5, max(0.0, fraction))
                motor_sum[i] += fraction
                if fraction > motor_max[i]:
                    motor_max[i] = fraction
                if fraction < thresholds.motor_saturation:
                    saturated = False
            throttle = sum(
                item.get(name, 0) - motor_low for name in motor_names
            ) / (len(motor_names) * motor_span)
            throttle = min(1.5, max(0.0, throttle))
            throttle_sum += throttle
            if throttle > stats.throttle_max:
                stats.throttle_max = throttle
            charts["throttle"].add(throttle)

            if saturated:
                stats.saturated_samples += 1
                saturation_run_us += dt_us
                if saturation_run_us >= thresholds.motor_saturation_ms * 1000:
                    _flag_once(
                        stats,
                        "motor-saturation",
                        "all motors above %d%% for %.0f ms at t=%.2fs"
                        % (
                            thresholds.motor_saturation * 100,
                            saturation_run_us / 1000,
                            _seconds(stats, time_us),
                        ),
                    )
            else:
                saturation_run_us = 0

        # battery
        if has_vbat:
            volts = _to_volts(item.get("vbatLatest", 0), header, vbat_adc)
            if stats.v_start is None:
                stats.v_start = volts
                stats.cells = _detect_cells(header, volts)
            stats.v_end = volts
            if stats.v_min is None or volts < stats.v_min:
                stats.v_min = volts
            charts["voltage"].add(volts)
        if has_amps:
            amps = item.get("amperageLatest", 0) / 100.0
            stats.amps_max = max(stats.amps_max, amps)
            amp_seconds += amps * dt_us / 1_000_000
            charts["current"].add(amps)

        # receiver
        if has_rssi:
            rssi = item.get("rssi", 0)
            if stats.rssi_min is None or rssi < stats.rssi_min:
                stats.rssi_min = rssi
            if rssi < stats.rssi_scale * thresholds.rssi_low:
                stats.low_rssi_samples += 1
                rssi_run_us += dt_us
                if rssi_run_us >= thresholds.rssi_low_ms * 1000:
                    _flag_once(
                        stats,
                        "low-rssi",
                        "RSSI fell to %d/%d for over %d ms at t=%.2fs"
                        % (
                            rssi,
                            int(stats.rssi_scale),
                            thresholds.rssi_low_ms,
                            _seconds(stats, time_us),
                        ),
                    )
            else:
                rssi_run_us = 0

        # gyro and accelerometer
        peak_dps = 0.0
        for i, name in enumerate(gyro_names):
            raw = item.get(name, 0)
            if abs(raw) >= GYRO_CLIP_RAW:
                stats.gyro_clipped += 1
            dps = abs(raw * scale)
            gyro_sq[i] += dps * dps
            if dps > gyro_peak[i]:
                gyro_peak[i] = dps
            peak_dps = max(peak_dps, dps)
        if gyro_names:
            charts["gyro"].add(peak_dps)

        if acc_names:
            magnitude = (
                math.sqrt(sum(item.get(name, 0) ** 2 for name in acc_names))
                / header.acc_1g
            )
            if magnitude > stats.acc_peak_g:
                stats.acc_peak_g = magnitude
            if (
                magnitude >= thresholds.crash_g
                and peak_dps >= thresholds.crash_gyro_dps
            ):
                _flag_once(
                    stats,
                    "impact",
                    "%.1f g with %.0f deg/s rotation at t=%.2fs -- looks like an impact"
                    % (magnitude, peak_dps, _seconds(stats, time_us)),
                )

    # -- wrap up ---------------------------------------------------------
    count = max(1, stats.main_frames)
    stats.motor_max = motor_max
    stats.motor_mean = [total / count for total in motor_sum]
    stats.throttle_mean = throttle_sum / count
    stats.gyro_peak = gyro_peak
    stats.gyro_rms = [math.sqrt(total / count) for total in gyro_sq]
    stats.mah_used = amp_seconds / 3.6
    stats.series = {name: buffer.samples for name, buffer in charts.items()}

    _final_checks(stats)
    return stats


# -- per frame-type handlers ---------------------------------------------
def _record_event(stats, event):
    entry = {"time": event.get("time", 0), "type": event.name}
    if event.name == "disarm":
        entry["reason"] = event.get("reasonName")
    elif event.name == "flightMode":
        entry["modes"] = decode_mode_flags(event.get("flags", 0))
    elif event.name in ("syncBeep", "loggingResume"):
        entry["time"] = event.get("time", entry["time"])
    stats.events.append(entry)
    stats.frame_counts["E"] = stats.frame_counts.get("E", 0) + 1


def _record_slow(stats, frame):
    """One receiver problem sets several of these flags, so report them once."""
    reasons = []
    phase = frame.get("failsafePhase", 0)
    if phase:
        reasons.append("failsafe phase %s" % FAILSAFE_PHASES.get(phase, phase))
    if frame.get("rxSignalReceived", 1) == 0:
        reasons.append("no signal")
    if frame.get("rxFlightChannelsValid", 1) == 0:
        reasons.append("flight channels invalid")
    if reasons:
        _flag_once(stats, "rx-loss", "receiver dropped out: " + ", ".join(reasons))


def _record_gps(stats, frame, home):
    sats = frame.get("GPS_numSat")
    if sats is not None:
        stats.gps_sats = sats if stats.gps_sats is None else max(stats.gps_sats, sats)
    speed = frame.get("GPS_speed")
    if speed is not None:  # centimetres per second
        kmh = speed * 0.036
        if stats.gps_speed_max_kmh is None or kmh > stats.gps_speed_max_kmh:
            stats.gps_speed_max_kmh = kmh
    altitude = frame.get("GPS_altitude")
    if altitude is not None and (
        stats.gps_alt_max is None or altitude > stats.gps_alt_max
    ):
        stats.gps_alt_max = altitude
    if home and frame.get("GPS_coord[0]") is not None:
        distance = _haversine(
            home[0] / 1e7,
            home[1] / 1e7,
            frame.get("GPS_coord[0]") / 1e7,
            frame.get("GPS_coord[1]") / 1e7,
        )
        if stats.gps_distance_max_m is None or distance > stats.gps_distance_max_m:
            stats.gps_distance_max_m = distance


def _final_checks(stats):
    """Rules that need the whole flight before they can fire."""
    thresholds = stats.thresholds

    if stats.dropout_pct > thresholds.dropout_pct:
        # Missing iterations mean one of two very different things, and
        # blaming the SD card for what was really corruption is unhelpful.
        if stats.decode.corrupt_frames:
            detail = (
                "%d of %d loop iterations are missing (%.1f%%), lost along with "
                "the corrupt frames below"
            )
        else:
            detail = (
                "%d of %d loop iterations never reached the log (%.1f%%) -- the "
                "logging device could not keep up"
            )
        _flag_once(
            stats,
            "logging-dropouts",
            detail
            % (stats.dropped_iterations, stats.expected_iterations, stats.dropout_pct),
        )

    if stats.decode.corrupt_frames:
        _flag_once(
            stats,
            "corruption",
            "%d frames were unreadable, %d bytes skipped over %d resyncs"
            % (
                stats.decode.corrupt_frames,
                stats.decode.corrupt_bytes,
                stats.decode.resyncs,
            ),
        )

    per_cell = stats.v_per_cell_min
    if per_cell is not None and per_cell < thresholds.cell_min_v:
        _flag_once(
            stats,
            "voltage-sag",
            "pack sagged to %.2f V/cell (%.2f V on %dS), below the %.2f V limit"
            % (per_cell, stats.v_min, stats.cells, thresholds.cell_min_v),
        )

    # A tumble pins one motor wide open for as long as the flight controller
    # fights it, which skews the whole-flight averages. Advising the pilot to
    # check prop balance after a crash is worse than saying nothing.
    crashed = any(anomaly["code"] == "impact" for anomaly in stats.anomalies)
    if stats.motor_imbalance > thresholds.motor_imbalance and not crashed:
        worst = stats.motor_mean.index(max(stats.motor_mean))
        _flag_once(
            stats,
            "motor-imbalance",
            "motor %d averaged %.0f%% while the quietest averaged %.0f%% -- check "
            "prop balance, bent shaft or a soft mount"
            % (
                worst + 1,
                100 * max(stats.motor_mean),
                100 * min(stats.motor_mean),
            ),
        )

    if stats.gyro_clipped:
        _flag_once(
            stats,
            "gyro-clipping",
            "%d gyro samples railed against the sensor limit" % stats.gyro_clipped,
        )

    for event in stats.events:
        if event.get("type") == "disarm" and event.get("reason") not in (
            "SWITCH",
            "STICKS",
            None,
        ):
            _flag_once(
                stats,
                "unexpected-disarm",
                "the flight controller disarmed itself: %s" % event["reason"],
            )

    if not stats.decode.reached_log_end:
        _flag_once(
            stats,
            "truncated",
            "the log has no end marker -- power was cut or the card was pulled "
            "while writing",
        )


# -- helpers --------------------------------------------------------------
def _flag_once(stats, code, message):
    for anomaly in stats.anomalies:
        if anomaly["code"] == code:
            return
    stats.anomalies.append({"code": code, "detail": message})


def _seconds(stats, time_us):
    start = stats.start_time_us or 0
    return (time_us - start) / 1_000_000


def _numbered_fields(header, base):
    """Collect ``base[0]``, ``base[1]``, ... in order, while they exist."""
    names = []
    index = 0
    while header.has_field("%s[%d]" % (base, index)):
        names.append("%s[%d]" % (base, index))
        index += 1
    return names


def _gyro_scale(header):
    """Degrees per second per raw gyro count.

    Betaflight 4.x converts the gyro to deg/s before logging it and writes a
    scale of exactly 1.0, so 1.0 is a valid value here, not a broken one.
    Falls back to a 2000 deg/s sensor on a 16-bit ADC, which is what almost
    every quad flies, when the header value is missing or implausible.
    """
    scale = header.gyro_scale
    if scale and 0.0001 < abs(scale) <= 1.0:
        return scale
    return 2000.0 / 32768.0


def _to_volts(raw, header, adc_mode):
    """Battery voltage in volts.

    Betaflight 4.x logs hundredths of a volt. Cleanflight-era logs stored the
    raw 12-bit ADC reading instead, which `adc_mode` converts with the
    reference decoder's formula.
    """
    if adc_mode:
        return raw * 330 * header.vbat_scale / 0xFFF / 10 / 1000.0
    return raw / 100.0


#: Betaflight's MAX_AUTO_DETECT_CELL_COUNT.
MAX_AUTO_DETECT_CELLS = 8

#: Fallback for the maximum per-cell voltage, in centivolts, when the log does
#: not carry a `vbatcellvoltage` header. Betaflight's default is 4.30 V.
DEFAULT_MAX_CELL_CENTIVOLTS = 430


def _cell_count(volts, max_cell_centivolts=DEFAULT_MAX_CELL_CENTIVOLTS):
    """Infer cell count exactly as Betaflight's autoDetectCellCount does.

    ``cells = voltage / vbatmaxcellvoltage + 1`` in integer centivolts. The
    firmware runs this the moment the pack is plugged in, which is why
    :func:`_detect_cells` feeds it the header's reference voltage rather than
    a live sample.
    """
    if volts is None or volts <= 0:
        return 0
    centivolts = int(round(volts * 100))
    cells = centivolts // max(1, max_cell_centivolts) + 1
    return min(cells, MAX_AUTO_DETECT_CELLS)


def _detect_cells(header, fallback_volts=None):
    """Work out the pack's cell count from the most reliable source available.

    ``vbatref`` is captured by the firmware when the battery is connected,
    before any sag, so it identifies the pack even when the log starts on a
    half-empty one. Inferring from the first logged sample instead misreads a
    6S pack that was already down to 21 V as 5S, and every per-cell figure
    after that is wrong.
    """
    limits = header.cell_voltage_limits
    max_cell = limits[2] if limits else DEFAULT_MAX_CELL_CENTIVOLTS

    reference = header.vbat_ref
    if reference:
        return _cell_count(reference / 100.0, max_cell)
    return _cell_count(fallback_volts, max_cell)


def _haversine(lat1, lon1, lat2, lon2):
    radius = 6371000.0
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return 2 * radius * math.asin(min(1.0, math.sqrt(a)))


def decode_mode_flags(flags):
    """Turn a flightModeFlags bitmask into a list of mode names."""
    modes = []
    for bit, name in enumerate(FLIGHT_MODES):
        if flags & (1 << bit):
            modes.append(name)
    remainder = flags >> len(FLIGHT_MODES)
    if remainder:
        modes.append("bits>%d" % len(FLIGHT_MODES))
    return modes


def describe_disarm(reason):
    return DISARM_REASONS.get(reason, "UNKNOWN(%s)" % reason)
