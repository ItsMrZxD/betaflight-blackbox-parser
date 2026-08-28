"""Decoder tests.

The first class is the important one: a log assembled byte by byte with the
expected values worked out by hand from the format description. Every other
test in this file leans on :mod:`postflight.synth`, which shares its idea of
the format with the decoder, so a mistake in the encodings could hide there.
This fixture cannot hide one.
"""

import unittest

from postflight.decoder import (
    MAX_ITERATION_JUMP,
    Event,
    Frame,
    Session,
    read_log,
    split_sessions,
)
from postflight.encoding import Writer
from postflight.errors import CorruptLog, UnsupportedLog
from postflight.synth import FlightModel, build_log, build_multi_session_log

HAND_BUILT_HEADER = (
    b"H Product:Blackbox flight data recorder by Nicholas Sherlock\n"
    b"H Data version:2\n"
    b"H I interval:32\n"
    b"H P interval:1/1\n"
    b"H Field I name:loopIteration,time,axisP[0],motor[0],motor[1]\n"
    b"H Field I signed:0,0,1,0,0\n"
    b"H Field I predictor:0,0,0,4,5\n"
    b"H Field I encoding:1,1,0,1,0\n"
    b"H Field P predictor:6,2,1,3,3\n"
    b"H Field P encoding:9,0,0,0,0\n"
    b"H minthrottle:1000\n"
    b"H maxthrottle:2000\n"
    b"H motorOutput:1000,2000\n"
    b"H vbatref:1600\n"
)

# Intra frame. Predictors are zero, zero, zero, minthrottle, motor[0]:
#   loopIteration  unsigned varint 0                     -> 0
#   time           unsigned varint 1000                  -> 1000
#   axisP[0]       signed varint -5 (zigzag 9)           -> -5
#   motor[0]       unsigned varint 200 + minthrottle     -> 1200
#   motor[1]       signed varint -50 + motor[0]          -> 1150
HAND_BUILT_I = bytes([0x49, 0x00, 0xE8, 0x07, 0x09, 0xC8, 0x01, 0x63])

# Inter frame. Predictors are increment, straight line, previous, average, average.
#   loopIteration  nothing encoded, previous + 1         -> 1
#   time           125 + (2*1000 - 1000)                 -> 1125
#   axisP[0]       2 + (-5)                              -> -3
#   motor[0]       10 + (1200 + 1200)/2                  -> 1210
#   motor[1]       -20 + (1150 + 1150)/2                 -> 1130
HAND_BUILT_P1 = bytes([0x50, 0xFA, 0x01, 0x04, 0x14, 0x27])

# Second inter frame, now with two different history entries to average.
#   time           0 + (2*1125 - 1000)                   -> 1250
#   axisP[0]       -1 + (-3)                             -> -4
#   motor[0]       5 + (1210 + 1200)/2                   -> 1210
#   motor[1]       0 + (1130 + 1150)/2                   -> 1140
HAND_BUILT_P2 = bytes([0x50, 0x00, 0x01, 0x0A, 0x00])

HAND_BUILT_END = b"E\xffEnd of log\x00"

HAND_BUILT_LOG = (
    HAND_BUILT_HEADER + HAND_BUILT_I + HAND_BUILT_P1 + HAND_BUILT_P2 + HAND_BUILT_END
)


class HandBuiltLogTest(unittest.TestCase):
    """Decode a log whose expected output was computed with pen and paper."""

    def setUp(self):
        self.session = Session(HAND_BUILT_LOG)
        self.frames = list(self.session.frames())

    def test_header_fields(self):
        header = self.session.header
        self.assertEqual(
            header.frames["I"].names,
            ["loopIteration", "time", "axisP[0]", "motor[0]", "motor[1]"],
        )
        self.assertEqual(header.min_throttle, 1000)
        self.assertEqual(header.motor_output_range, (1000, 2000))
        self.assertEqual(header.vbat_ref, 1600)
        self.assertTrue(header.frames["I"].fields[2].signed)
        self.assertFalse(header.frames["I"].fields[3].signed)
        # The inter frame borrows its names and signs from the intra frame.
        self.assertEqual(header.frames["P"].names, header.frames["I"].names)

    def test_intra_frame_values(self):
        frame = self.frames[0]
        self.assertEqual(frame.kind, "I")
        self.assertEqual(frame.values, [0, 1000, -5, 1200, 1150])

    def test_first_inter_frame_values(self):
        frame = self.frames[1]
        self.assertEqual(frame.kind, "P")
        self.assertEqual(frame.values, [1, 1125, -3, 1210, 1130])

    def test_second_inter_frame_values(self):
        self.assertEqual(self.frames[2].values, [2, 1250, -4, 1210, 1140])

    def test_log_end_is_recognised(self):
        self.assertEqual(len(self.frames), 3)
        self.assertTrue(self.session.decode_stats.reached_log_end)
        self.assertEqual(self.session.decode_stats.corrupt_frames, 0)

    def test_field_access_by_name(self):
        frame = self.frames[1]
        self.assertEqual(frame["time"], 1125)
        self.assertEqual(frame.get("motor[0]"), 1210)
        self.assertIsNone(frame.get("nosuchfield"))
        self.assertIn("axisP[0]", frame)
        self.assertEqual(frame.as_dict()["motor[1]"], 1130)

    def test_increment_predictor_honours_the_logging_rate(self):
        """Regression: Betaflight 4.2 writes a bare "P interval:16".

        It means one inter frame per sixteen loop iterations, so the counter
        has to jump by sixteen. Reading it as 1/1 made every real log look
        like it had dropped 94% of its frames.
        """
        log = HAND_BUILT_LOG.replace(b"H P interval:1/1\n", b"H P interval:16\n")
        session = Session(log)
        self.assertEqual(session.header.p_interval, (1, 16))
        iterations = [
            frame["loopIteration"]
            for frame in session.frames()
            if frame.kind in ("I", "P")
        ]
        self.assertEqual(iterations, [0, 16, 32])

    def test_iterating_twice_gives_the_same_answer(self):
        again = [frame.values for frame in self.session.frames()]
        self.assertEqual(again, [frame.values for frame in self.frames])
        self.assertEqual(self.session.decode_stats.frame_counts["P"], 2)


def make_intra(iteration, time_us, axis_p=0, motor0_residual=200, motor1_residual=0):
    """Build a valid intra frame for HAND_BUILT_HEADER's five columns."""
    writer = Writer()
    writer.write_byte(ord("I"))
    writer.write_unsigned_vb(iteration)
    writer.write_unsigned_vb(time_us)
    writer.write_signed_vb(axis_p)
    writer.write_unsigned_vb(motor0_residual)
    writer.write_signed_vb(motor1_residual)
    return writer.bytes()


class LogEndMarkerTest(unittest.TestCase):
    """The end-of-log event must carry the literal marker to be believed.

    Betaflight writes ``"End of log"`` plus a NUL after the 0xFF event byte
    (blackbox.c, FLIGHT_LOG_EVENT_LOG_END). Accepting the event byte alone let
    any stray 0x45 0xFF pair in damaged data end the session early while
    reporting a clean finish and no corruption.
    """

    def test_a_valid_marker_ends_the_log(self):
        session = Session(HAND_BUILT_LOG)
        frames = list(session.frames())
        self.assertEqual(len(frames), 3)
        self.assertTrue(session.decode_stats.reached_log_end)
        self.assertEqual(session.decode_stats.corrupt_frames, 0)

    def test_a_malformed_marker_is_corruption_not_an_ending(self):
        log = HAND_BUILT_LOG.replace(HAND_BUILT_END, b"E\xffNot the end!")
        session = Session(log)
        list(session.frames())
        self.assertFalse(session.decode_stats.reached_log_end)
        self.assertGreater(session.decode_stats.corrupt_frames, 0)

    def test_a_truncated_marker_is_corruption(self):
        log = HAND_BUILT_LOG.replace(HAND_BUILT_END, b"E\xffEnd of")
        session = Session(log)
        list(session.frames())
        self.assertFalse(session.decode_stats.reached_log_end)
        self.assertGreater(session.decode_stats.corrupt_frames, 0)

    def test_real_frames_before_a_bogus_marker_are_still_returned(self):
        log = HAND_BUILT_LOG.replace(HAND_BUILT_END, b"E\xffnope")
        frames = [f for f in Session(log).frames() if f.kind in ("I", "P")]
        self.assertEqual([f.values[0] for f in frames], [0, 1, 2])

    def test_stray_event_bytes_cannot_silently_truncate_a_flight(self):
        """The exact adversarial case from the audit.

        Injecting 0x45 0xFF into the middle of a clean log used to yield about
        half the frames with corrupt_frames = 0 and reached_log_end = True, so
        the report showed no warnings at all.
        """
        good = build_log(duration=1.0, loop_hz=1000)
        clean = Session(good)
        clean_frames = sum(1 for f in clean.frames() if f.kind in ("I", "P"))

        header_end = Session(good).header.end_offset
        midpoint = header_end + (len(good) - header_end) // 2
        spliced = good[:midpoint] + b"\x45\xff" + good[midpoint:]

        session = Session(spliced)
        frames = sum(1 for f in session.frames() if f.kind in ("I", "P"))
        stats = session.decode_stats

        silently_truncated = (
            frames < clean_frames
            and stats.corrupt_frames == 0
            and stats.reached_log_end
        )
        self.assertFalse(
            silently_truncated,
            "lost %d of %d frames while reporting a clean, uncorrupted log"
            % (clean_frames - frames, clean_frames),
        )
        # It should also recover most of the flight rather than stopping dead.
        self.assertGreater(frames, clean_frames * 0.9)


class TimestampRolloverTest(unittest.TestCase):
    """`micros()` wraps every 71.6 minutes of flight-controller uptime.

    Follows the reference decoder, which detects a backwards step that is
    small in modulo-32-bit terms and adds 2**32 to an accumulator.
    """

    WRAP = 1 << 32

    def _log(self, *frames):
        return HAND_BUILT_HEADER + b"".join(frames) + HAND_BUILT_END

    def test_frames_after_the_wrap_survive(self):
        log = self._log(
            make_intra(0, self.WRAP - 1_000_000),
            make_intra(1, self.WRAP - 500_000),
            make_intra(2, 250_000),  # counter has wrapped
            make_intra(3, 750_000),
        )
        session = Session(log)
        frames = [f for f in session.frames() if f.kind in ("I", "P")]
        self.assertEqual(len(frames), 4, "frames after the wrap were discarded")
        self.assertEqual(session.decode_stats.corrupt_frames, 0)

    def test_time_is_reported_monotonically_across_the_wrap(self):
        log = self._log(
            make_intra(0, self.WRAP - 1_000_000),
            make_intra(1, 250_000),
        )
        times = [f["time"] for f in Session(log).frames() if f.kind in ("I", "P")]
        self.assertEqual(times, [self.WRAP - 1_000_000, self.WRAP + 250_000])
        self.assertLess(times[0], times[1])

    def test_the_elapsed_time_across_a_wrap_is_correct(self):
        log = self._log(
            make_intra(0, self.WRAP - 1_000_000),
            make_intra(1, 250_000),
        )
        stats = Session(log).stats()
        self.assertAlmostEqual(stats.duration_s, 1.25, places=6)

    def test_a_genuine_backwards_jump_is_still_rejected(self):
        # Two seconds backwards is far too big to be a wrap, so it is damage.
        log = self._log(
            make_intra(0, 60_000_000),
            make_intra(1, 58_000_000),
        )
        session = Session(log)
        frames = [f for f in session.frames() if f.kind in ("I", "P")]
        self.assertEqual(len(frames), 1)
        self.assertGreater(session.decode_stats.corrupt_frames, 0)

    def test_an_implausible_forward_jump_is_still_rejected(self):
        log = self._log(
            make_intra(0, 1_000_000),
            make_intra(1, 1_000_000 + 20_000_000),  # 20 s gap, over the limit
        )
        session = Session(log)
        frames = [f for f in session.frames() if f.kind in ("I", "P")]
        self.assertEqual(len(frames), 1)
        self.assertGreater(session.decode_stats.corrupt_frames, 0)

    def test_exactly_at_the_boundary(self):
        log = self._log(make_intra(0, self.WRAP - 1), make_intra(1, 0))
        times = [f["time"] for f in Session(log).frames() if f.kind in ("I", "P")]
        self.assertEqual(times, [self.WRAP - 1, self.WRAP])


class IterationJumpLimitTest(unittest.TestCase):
    """The iteration sanity bound must stay at the reference decoder's value.

    blackbox-tools uses MAXIMUM_ITERATION_JUMP_BETWEEN_FRAMES = 500 * 10. This
    project once used 500 * 256, twenty-six times looser, which let more
    garbage survive validation during resynchronisation. The bound still has
    to clear the largest legitimate gap: one intra-frame interval times the
    inter-frame denominator, 256 * 16 = 4096 on the real logs tested.
    """

    def _log(self, jump):
        return (
            HAND_BUILT_HEADER
            + make_intra(0, 1_000_000)
            + make_intra(jump, 1_001_000)
            + HAND_BUILT_END
        )

    def _accepted(self, jump):
        session = Session(self._log(jump))
        frames = [f for f in session.frames() if f.kind in ("I", "P")]
        return len(frames) == 2

    def test_the_bound_is_the_reference_value(self):
        self.assertEqual(MAX_ITERATION_JUMP, 500 * 10)

    def test_a_realistic_worst_case_gap_is_accepted(self):
        # 256 intra-frame interval x 16 inter-frame denominator.
        self.assertTrue(self._accepted(4096))

    def test_a_gap_just_under_the_bound_is_accepted(self):
        self.assertTrue(self._accepted(MAX_ITERATION_JUMP - 1))

    def test_a_gap_at_the_bound_is_rejected(self):
        self.assertFalse(self._accepted(MAX_ITERATION_JUMP))

    def test_a_far_larger_gap_is_rejected(self):
        session = Session(self._log(500 * 256))
        frames = [f for f in session.frames() if f.kind in ("I", "P")]
        self.assertEqual(len(frames), 1)
        self.assertGreater(session.decode_stats.corrupt_frames, 0)


class DataVersionTest(unittest.TestCase):
    """TAG8_4S16 has two layouts and the header picks one.

    Reference: parser.c chooses streamReadTag8_4S16_v1 when dataVersion < 2.
    Decoding a v1 log with v2 rules yields wrong numbers, not an error, so
    this has to be driven by the header rather than assumed.
    """

    def _log(self, version):
        header = (
            HAND_BUILT_HEADER.replace(b"H Data version:2\n", b"H Data version:%d\n" % version)
            .replace(
                b"H Field I encoding:1,1,0,1,0\n", b"H Field I encoding:1,1,8,8,8\n"
            )
            .replace(b"H Field I predictor:0,0,0,4,5\n", b"H Field I predictor:0,0,0,0,0\n")
        )
        # loopIteration, time, then one TAG8_4S16 group covering the last three
        # columns plus one field the group overruns into is not allowed, so the
        # frame is: iteration, time, and a 4-value group over axisP/motor0/motor1
        # -- the fourth value of the group would overrun, so pad the field list.
        header = header.replace(
            b"H Field I name:loopIteration,time,axisP[0],motor[0],motor[1]\n",
            b"H Field I name:loopIteration,time,a,b,c,d\n",
        )
        header = header.replace(b"H Field I signed:0,0,1,0,0\n", b"H Field I signed:0,0,1,1,1,1\n")
        header = header.replace(
            b"H Field I encoding:1,1,8,8,8\n", b"H Field I encoding:1,1,8,8,8,8\n"
        )
        header = header.replace(
            b"H Field I predictor:0,0,0,0,0\n", b"H Field I predictor:0,0,0,0,0,0\n"
        )
        header = header.replace(b"H Field P predictor:6,2,1,3,3\n", b"H Field P predictor:6,2,1,1,1,1\n")
        header = header.replace(b"H Field P encoding:9,0,0,0,0\n", b"H Field P encoding:9,0,0,0,0,0\n")

        frame = Writer()
        frame.write_byte(ord("I"))
        frame.write_unsigned_vb(0)
        frame.write_unsigned_vb(1000)
        frame.write_raw(bytes([0x55, 0x12, 0x34]))  # the golden nibble group
        return header + frame.bytes() + HAND_BUILT_END

    def test_version_2_uses_the_v2_layout(self):
        frame = next(iter(Session(self._log(2)).frames()))
        self.assertEqual(frame.values[2:], [1, 2, 3, 4])

    def test_version_1_uses_the_v1_layout(self):
        frame = next(iter(Session(self._log(1)).frames()))
        self.assertEqual(frame.values[2:], [2, 1, 4, 3])

    def test_the_versions_really_differ(self):
        v1 = next(iter(Session(self._log(1)).frames())).values
        v2 = next(iter(Session(self._log(2)).frames())).values
        self.assertNotEqual(v1, v2)


class UnsupportedEncodingTest(unittest.TestCase):
    def test_tag2_3svariable_is_refused_rather_than_guessed(self):
        log = HAND_BUILT_LOG.replace(
            b"H Field P encoding:9,0,0,0,0\n", b"H Field P encoding:9,10,10,10,0\n"
        )
        with self.assertRaises(UnsupportedLog) as caught:
            list(Session(log).frames())
        self.assertIn("TAG2_3SVARIABLE", str(caught.exception))


class MalformedLoggingRateTest(unittest.TestCase):
    def test_a_zero_numerator_does_not_destroy_the_log(self):
        """`P interval:0/2` is not something the firmware writes.

        Taken literally it makes every inter frame appear to skip 128000
        iterations, so all of them fail validation and the flight is lost.
        """
        log = HAND_BUILT_LOG.replace(b"H P interval:1/1\n", b"H P interval:0/2\n")
        session = Session(log)
        frames = [f for f in session.frames() if f.kind in ("I", "P")]
        self.assertEqual(len(frames), 3)
        self.assertEqual(session.decode_stats.corrupt_frames, 0)

    def test_a_numerator_above_the_denominator_is_ignored(self):
        log = HAND_BUILT_LOG.replace(b"H P interval:1/1\n", b"H P interval:5/2\n")
        frames = [f for f in Session(log).frames() if f.kind in ("I", "P")]
        self.assertEqual(len(frames), 3)


class UnknownEventTest(unittest.TestCase):
    """Betaflight only emits six event types; anything else is damage.

    Verified against blackboxLogEvent in blackbox.c, which writes SYNC_BEEP,
    FLIGHTMODE, DISARM, INFLIGHT_ADJUSTMENT, LOGGING_RESUME and LOG_END. The
    reference decoder assumes an unknown event has no payload and keeps
    reading, which can desynchronise; this decoder treats it as corruption and
    resynchronises on a verified intra frame instead.
    """

    def test_an_unknown_event_is_treated_as_corruption(self):
        log = HAND_BUILT_HEADER + HAND_BUILT_I + b"E\x28" + HAND_BUILT_P1 + HAND_BUILT_END
        session = Session(log)
        frames = [f for f in session.frames() if f.kind in ("I", "P")]
        self.assertGreaterEqual(len(frames), 1)
        self.assertGreater(session.decode_stats.corrupt_frames, 0)

    def test_every_event_betaflight_writes_is_decoded(self):
        writer = Writer()
        for code, payload in (
            (0, [12345]),  # sync beep
            (30, [3, 0]),  # flight mode
            (14, [7, 999]),  # logging resume
            (15, [4]),  # disarm
        ):
            writer.write_byte(ord("E"))
            writer.write_byte(code)
            for value in payload:
                writer.write_unsigned_vb(value)
        log = HAND_BUILT_HEADER + HAND_BUILT_I + writer.bytes() + HAND_BUILT_END
        session = Session(log)
        events = [f for f in session.frames() if isinstance(f, Event)]
        self.assertEqual(
            [e.name for e in events],
            ["syncBeep", "flightMode", "loggingResume", "disarm"],
        )
        self.assertEqual(session.decode_stats.corrupt_frames, 0)
        self.assertTrue(session.decode_stats.reached_log_end)

    def test_inflight_adjustment_both_payload_shapes(self):
        import struct as _struct

        writer = Writer()
        writer.write_byte(ord("E"))
        writer.write_byte(13)
        writer.write_byte(5)  # integer form, function 5
        writer.write_signed_vb(-42)
        writer.write_byte(ord("E"))
        writer.write_byte(13)
        writer.write_byte(5 | 0x80)  # float form
        writer.write_raw(_struct.pack("<f", 1.5))
        log = HAND_BUILT_HEADER + HAND_BUILT_I + writer.bytes() + HAND_BUILT_END
        events = [f for f in Session(log).frames() if isinstance(f, Event)]
        self.assertEqual([e.get("value") for e in events], [-42, 1.5])


class BadHeaderTest(unittest.TestCase):
    def test_no_intra_frame_definition(self):
        header = HAND_BUILT_HEADER.replace(b"H Field I predictor:0,0,0,4,5\n", b"")
        with self.assertRaises(CorruptLog):
            Session(header + HAND_BUILT_END)

    def test_field_count_mismatch(self):
        header = HAND_BUILT_HEADER.replace(
            b"H Field I encoding:1,1,0,1,0\n", b"H Field I encoding:1,1,0\n"
        )
        with self.assertRaises(CorruptLog):
            Session(header + HAND_BUILT_END)

    def test_unterminated_header_line(self):
        with self.assertRaises(CorruptLog):
            Session(b"H Product:Blackbox flight data recorder")

    def test_not_a_log_at_all(self):
        with self.assertRaises(CorruptLog):
            Session(b"this is a jpeg, honest")


class RoundTripTest(unittest.TestCase):
    """Every value the flight model produced must survive encode and decode."""

    def test_all_fields_match_the_flight_that_was_generated(self):
        duration, loop_hz, seed = 0.5, 1000, 11
        data = build_log(duration=duration, loop_hz=loop_hz, seed=seed, gps=True)
        session = Session(data)

        model = FlightModel(duration, loop_hz, seed)
        expected = [model.sample(i) for i in range(model.iterations)]

        decoded = [f for f in session.frames() if f.kind in ("I", "P")]
        self.assertEqual(len(decoded), len(expected))
        for frame, row in zip(decoded, expected):
            self.assertEqual(frame.as_dict(), row)

    def test_both_frame_kinds_are_used(self):
        session = Session(build_log(duration=0.2, i_interval=32))
        counts = {}
        for frame in session.frames():
            counts[frame.kind] = counts.get(frame.kind, 0) + 1
        self.assertEqual(counts["I"], 200 // 32 + 1)
        self.assertEqual(counts["I"] + counts["P"], 200)

    def test_slow_and_event_frames_decode(self):
        session = Session(build_log(duration=0.5, slow_interval=100, rx_glitch_at=0.2))
        slow = [f for f in session.frames() if f.kind == "S"]
        self.assertTrue(slow)
        self.assertEqual(slow[0]["flightModeFlags"], 0b11)
        self.assertTrue(any(frame["failsafePhase"] == 1 for frame in slow))

        events = [f for f in session.frames() if isinstance(f, Event)]
        names = [event.name for event in events]
        self.assertIn("flightMode", names)
        self.assertIn("disarm", names)
        disarm = events[names.index("disarm")]
        self.assertEqual(disarm.get("reasonName"), "SWITCH")

    def test_gps_frames_use_the_home_position(self):
        session = Session(build_log(duration=1.0, gps=True))
        gps = [f for f in session.frames() if f.kind == "G"]
        self.assertTrue(gps)
        # Home is 51.4, -0.1 degrees; every fix should be within a few
        # hundred metres of it, which the home-coordinate predictor guarantees.
        for frame in gps:
            self.assertAlmostEqual(frame["GPS_coord[0]"] / 1e7, 51.4, places=2)
            self.assertAlmostEqual(frame["GPS_coord[1]"] / 1e7, -0.1, places=2)
        self.assertEqual(gps[0]["GPS_numSat"], 11)

    def test_gps_time_tracks_the_last_main_frame(self):
        session = Session(build_log(duration=1.0, gps=True))
        last_main_time = 0
        for frame in session.frames():
            if frame.kind in ("I", "P"):
                last_main_time = frame["time"]
            elif frame.kind == "G":
                self.assertEqual(frame["time"], last_main_time)


class MultipleSessionTest(unittest.TestCase):
    def test_split_finds_every_session(self):
        data = build_multi_session_log(3, duration=0.2)
        bounds = split_sessions(data)
        self.assertEqual(len(bounds), 3)
        self.assertEqual(bounds[0][0], 0)
        self.assertEqual(bounds[-1][1], len(data))
        for start, end in bounds:
            self.assertTrue(data[start:end].startswith(b"H Product:Blackbox"))

    def test_no_sessions_in_junk(self):
        self.assertEqual(split_sessions(b"nothing to see here"), [])

    def test_read_log_returns_one_session_per_flight(self):
        import os
        import tempfile

        data = build_multi_session_log(2, duration=0.2)
        handle, path = tempfile.mkstemp(suffix=".bbl")
        try:
            with os.fdopen(handle, "wb") as out:
                out.write(data)
            sessions = read_log(path)
            self.assertEqual(len(sessions), 2)
            self.assertEqual([s.index for s in sessions], [0, 1])
            self.assertEqual(sessions[1].offset, split_sessions(data)[1][0])
            for session in sessions:
                self.assertTrue(list(session.frames()))
        finally:
            os.unlink(path)


class CorruptionTest(unittest.TestCase):
    def test_decoder_recovers_from_scattered_damage(self):
        clean = Session(build_log(duration=2.0, seed=5))
        clean_count = sum(1 for f in clean.frames() if f.kind in ("I", "P"))

        damaged = Session(build_log(duration=2.0, seed=5, corrupt=20))
        frames = [f for f in damaged.frames() if f.kind in ("I", "P")]

        stats = damaged.decode_stats
        self.assertGreater(stats.resyncs, 0, "expected the decoder to resynchronise")
        self.assertGreater(
            len(frames), clean_count * 0.75, "lost too much of the log to 20 bad bytes"
        )
        self.assertLess(len(frames), clean_count, "corruption should cost some frames")

    def test_frames_after_a_resync_are_still_sane(self):
        session = Session(build_log(duration=2.0, seed=9, corrupt=15))
        previous_time = None
        for frame in session.frames():
            if frame.kind not in ("I", "P"):
                continue
            time = frame["time"]
            if previous_time is not None:
                self.assertGreaterEqual(time, previous_time)
                self.assertLess(time - previous_time, 10_000_000)
            previous_time = time

    def test_truncated_log_has_no_end_marker(self):
        session = Session(build_log(duration=0.3, end_cleanly=False))
        list(session.frames())
        self.assertFalse(session.decode_stats.reached_log_end)

    def test_log_cut_off_mid_frame(self):
        data = build_log(duration=0.5)
        session = Session(data[: len(data) - 400])
        frames = list(session.frames())
        self.assertTrue(frames, "a truncated log should still yield its good frames")
        self.assertFalse(session.decode_stats.reached_log_end)


class FrameHelperTest(unittest.TestCase):
    def test_repr_is_informative(self):
        frame = Frame("I", [1, 2], {"a": 0, "b": 1}, 0)
        self.assertIn("'I'", repr(frame))
        event = Event(15, "disarm", {"reason": 4}, 0)
        self.assertIn("disarm", repr(event))
        self.assertEqual(event.kind, "E")
        self.assertEqual(event.get("missing", "fallback"), "fallback")


if __name__ == "__main__":
    unittest.main()
