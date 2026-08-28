"""Flight statistics and the anomaly rules.

The rule tests come in pairs on purpose: a flight that should trip a warning
and a clean flight that should not. A detector that fires on everything is
indistinguishable from a broken one.
"""

import json
import unittest

from postflight.decoder import Session
from postflight.encoding import Writer
from postflight.stats import (
    _cell_count,
    _detect_cells,
    _final_checks,
    _gyro_scale,
    _haversine,
    _to_volts,
    compute_stats,
    decode_mode_flags,
)
from postflight.synth import build_log
from tests.test_decoder import HAND_BUILT_LOG


def codes(stats):
    return {anomaly["code"] for anomaly in stats.anomalies}


def _battery_log(vbatref, samples):
    """A minimal log carrying an absolute vbatLatest column.

    Hand-assembled rather than generated, so the header reference and the
    logged samples can disagree the way they do on a pack that was already
    part-used when logging started.
    """
    header = (
        b"H Product:Blackbox flight data recorder by Nicholas Sherlock\n"
        b"H Data version:2\n"
        b"H I interval:32\n"
        b"H P interval:1/1\n"
        b"H Field I name:loopIteration,time,vbatLatest\n"
        b"H Field I signed:0,0,0\n"
        b"H Field I predictor:0,0,0\n"
        b"H Field I encoding:1,1,1\n"
        b"H Field P predictor:6,2,1\n"
        b"H Field P encoding:9,0,0\n"
        b"H vbatref:%d\n" % vbatref
        + b"H vbatcellvoltage:330,350,430\n"
    )
    body = Writer()
    for iteration, (time_us, centivolts) in enumerate(samples):
        body.write_byte(ord("I"))
        body.write_unsigned_vb(iteration)
        body.write_unsigned_vb(time_us)
        body.write_unsigned_vb(centivolts)
    return header + body.bytes() + b"E\xffEnd of log\x00"


#: 6S pack, reference 25.2 V at plug-in, already down to 21.0 V when armed.
BATTERY_LOG = _battery_log(2520, [(0, 2100), (1000, 2090)])


class CleanFlightTest(unittest.TestCase):
    """A tidy two second hover should raise no warnings whatsoever."""

    @classmethod
    def setUpClass(cls):
        cls.session = Session(build_log(duration=2.0, seed=3, gps=True))
        cls.stats = cls.session.stats()

    def test_timing(self):
        self.assertAlmostEqual(self.stats.duration_s, 1.999, places=3)
        self.assertAlmostEqual(self.stats.loop_hz, 1000, delta=1)
        self.assertEqual(self.stats.main_frames, 2000)
        self.assertEqual(self.stats.expected_iterations, 2000)
        self.assertEqual(self.stats.dropped_iterations, 0)
        self.assertEqual(self.stats.dropout_pct, 0.0)

    def test_frame_counts_add_up(self):
        counts = self.stats.frame_counts
        self.assertEqual(counts["I"] + counts["P"], 2000)
        self.assertGreater(counts["G"], 0)
        self.assertGreater(counts["S"], 0)
        self.assertGreater(counts["E"], 0)

    def test_battery(self):
        self.assertEqual(self.stats.cells, 4)
        self.assertAlmostEqual(self.stats.v_start, 16.8, delta=0.3)
        self.assertLess(self.stats.v_min, self.stats.v_start)
        self.assertGreater(self.stats.v_per_cell_min, 3.3)
        self.assertGreater(self.stats.amps_max, 1.0)
        self.assertGreater(self.stats.mah_used, 0.0)

    def test_motors_and_throttle(self):
        self.assertEqual(len(self.stats.motor_max), 4)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in self.stats.motor_max))
        self.assertLessEqual(self.stats.throttle_mean, self.stats.throttle_max)
        self.assertLess(self.stats.motor_imbalance, 0.12)

    def test_gyro_is_reported_in_degrees_per_second(self):
        # The model flies at a few hundred deg/s; if the scale were being
        # ignored these would be in the tens of thousands.
        self.assertTrue(all(10 < peak < 2000 for peak in self.stats.gyro_peak))
        self.assertTrue(all(peak > 0 for peak in self.stats.gyro_rms))
        self.assertEqual(self.stats.gyro_clipped, 0)

    def test_gps(self):
        self.assertEqual(self.stats.gps_sats, 11)
        self.assertGreater(self.stats.gps_speed_max_kmh, 0)
        self.assertIsNotNone(self.stats.gps_distance_max_m)

    def test_events(self):
        types = [event["type"] for event in self.stats.events]
        self.assertIn("flightMode", types)
        self.assertIn("disarm", types)
        disarm = self.stats.events[types.index("disarm")]
        self.assertEqual(disarm["reason"], "SWITCH")

    def test_no_false_positives(self):
        self.assertEqual(codes(self.stats), set(), "clean flight raised warnings")

    def test_charts_are_populated_and_bounded(self):
        for name in ("throttle", "voltage", "current", "gyro"):
            series = self.stats.series[name]
            self.assertTrue(series, "no samples for %s" % name)
            self.assertLessEqual(len(series), 2048)

    def test_as_dict_is_json_serialisable(self):
        payload = json.loads(json.dumps(self.stats.as_dict()))
        self.assertEqual(payload["cells"], 4)
        self.assertEqual(payload["mainFrames"], 2000)
        self.assertEqual(payload["anomalies"], [])
        self.assertIn("gyroPeakDps", payload)


class AnomalyTest(unittest.TestCase):
    def test_impact_is_detected(self):
        session = Session(build_log(duration=4.0, seed=4, crash_at=3.0))
        stats = session.stats()
        self.assertIn("impact", codes(stats))
        detail = [a for a in stats.anomalies if a["code"] == "impact"][0]["detail"]
        self.assertIn("g with", detail)

    def test_receiver_dropout_is_detected(self):
        session = Session(build_log(duration=4.0, seed=4, rx_glitch_at=1.5))
        found = codes(session.stats())
        self.assertIn("rx-loss", found)
        self.assertIn("low-rssi", found)

    def test_receiver_dropout_is_reported_once(self):
        session = Session(build_log(duration=4.0, seed=4, rx_glitch_at=1.5))
        anomalies = [a for a in session.stats().anomalies if a["code"] == "rx-loss"]
        self.assertEqual(len(anomalies), 1)

    def test_truncated_log_is_flagged(self):
        session = Session(build_log(duration=1.0, end_cleanly=False))
        self.assertIn("truncated", codes(session.stats()))

    def test_corruption_is_flagged_and_explains_the_missing_iterations(self):
        session = Session(build_log(duration=2.0, seed=5, corrupt=20))
        stats = session.stats()
        found = codes(stats)
        self.assertIn("corruption", found)
        if "logging-dropouts" in found:
            detail = [
                a for a in stats.anomalies if a["code"] == "logging-dropouts"
            ][0]["detail"]
            self.assertIn("corrupt frames", detail)

    def _skewed(self):
        """A finished flight whose motor averages are badly lopsided."""
        stats = Session(build_log(duration=1.0, seed=3)).stats()
        stats.motor_mean = [0.60, 0.20, 0.21, 0.22]
        stats.anomalies = []
        return stats

    def test_imbalance_is_reported_on_an_ordinary_flight(self):
        stats = self._skewed()
        _final_checks(stats)
        self.assertIn("motor-imbalance", codes(stats))

    def test_a_crash_does_not_produce_prop_balance_advice(self):
        """A tumble pins one motor open and skews the whole-flight average.

        Telling the pilot to check for a bent shaft when the real event was an
        impact sends them after the wrong fault. The pair of tests above and
        below fix the rule in both directions: same lopsided averages, one
        with an impact recorded and one without.
        """
        stats = self._skewed()
        stats.anomalies = [{"code": "impact", "detail": "9.0 g at t=1.00s"}]
        _final_checks(stats)
        self.assertNotIn("motor-imbalance", codes(stats))

    def test_a_real_crash_flight_gives_no_prop_balance_advice(self):
        session = Session(build_log(duration=4.0, seed=4, crash_at=3.0))
        found = codes(session.stats())
        self.assertIn("impact", found)
        self.assertNotIn("motor-imbalance", found)

    def test_thresholds_can_be_argued_with(self):
        session = Session(build_log(duration=2.0, seed=3))
        stats = compute_stats(session)
        self.assertEqual(stats.thresholds.crash_g, 6.0)
        self.assertEqual(stats.thresholds.cell_min_v, 3.30)


class ConversionTest(unittest.TestCase):
    def test_cell_count_detection(self):
        self.assertEqual(_cell_count(16.8), 4)
        self.assertEqual(_cell_count(25.2), 6)
        self.assertEqual(_cell_count(4.2), 1)
        self.assertEqual(_cell_count(12.6), 3)

    def test_cell_count_matches_betaflight_across_realistic_packs(self):
        """Same rule as autoDetectCellCount: volts / vbatmaxcellvoltage + 1.

        Checked at both ends of each pack's usable range, using a fresh
        reference voltage as the firmware does when the battery is plugged in.
        """
        for volts, cells in [
            (4.20, 1), (3.30, 1),
            (8.40, 2), (6.60, 2),
            (12.60, 3), (9.90, 3),
            (16.80, 4), (13.20, 4),
            (21.00, 5),
            (25.20, 6),
        ]:
            self.assertEqual(_cell_count(volts), cells, "%.2f V" % volts)

    def test_a_discharged_pack_is_genuinely_ambiguous_from_voltage_alone(self):
        """Documents a limitation the firmware shares, and how it is avoided.

        16.5 V is a flat 5S and a healthy 4S; 19.8 V is a flat 6S and a
        healthy 5S. No rule can separate them from one live sample, which is
        why cell count comes from the header's plug-in reference instead.
        """
        self.assertEqual(_cell_count(16.5), 4)  # really 5S, flat
        self.assertEqual(_cell_count(19.8), 5)  # really 6S, flat

        log = HAND_BUILT_LOG.replace(b"H vbatref:1600\n", b"H vbatref:2520\n")
        header = Session(log).header
        self.assertEqual(_detect_cells(header, fallback_volts=19.8), 6)

    def test_cell_count_respects_the_headers_max_cell_voltage(self):
        # An HV pack profile raises the ceiling to 4.35 V per cell, which
        # changes the answer for voltages that straddle the two limits.
        self.assertEqual(_cell_count(17.3, 435), 4)
        self.assertEqual(_cell_count(17.3, 430), 5)

    def test_cell_count_is_capped_and_handles_no_battery(self):
        self.assertEqual(_cell_count(0), 0)
        self.assertEqual(_cell_count(None), 0)
        self.assertEqual(_cell_count(1000.0), 8)


class CellCountFromHeaderTest(unittest.TestCase):
    """Cell count must come from the header reference, not the first sample.

    `vbatref` is captured when the pack is connected, before any sag. A 6S
    pack that is already down to 21.0 V when logging starts reads as 5S if the
    first live sample is used, and every per-cell figure after that is wrong.
    """

    def _header(self, vbatref):
        log = HAND_BUILT_LOG.replace(b"H vbatref:1600\n", b"H vbatref:%d\n" % vbatref)
        return Session(log).header

    def test_a_discharged_six_cell_pack_is_not_mistaken_for_five(self):
        header = self._header(2520)  # 25.2 V measured at plug-in
        self.assertEqual(_detect_cells(header, fallback_volts=21.0), 6)
        # The first live sample alone would have said 5.
        self.assertEqual(_cell_count(21.0), 5)

    def test_fully_charged_packs_are_unaffected(self):
        for reference, cells in [(1680, 4), (2520, 6), (840, 2), (1260, 3)]:
            self.assertEqual(_detect_cells(self._header(reference)), cells)

    def test_a_full_report_uses_the_reference_not_the_first_sample(self):
        """End to end, not just the helper.

        A 6S pack whose header reference is 25.2 V but whose first logged
        sample is 21.0 V must still be reported as 6S, with per-cell figures
        and the sag warning computed against six cells.
        """
        stats = Session(BATTERY_LOG).stats()
        self.assertAlmostEqual(stats.v_start, 21.00, places=2)
        self.assertEqual(stats.cells, 6, "fell back to the first live sample")
        self.assertAlmostEqual(stats.v_per_cell_min, 20.90 / 6, places=4)
        # 3.48 V/cell on 6S is fine; on the mistaken 5S it would read 4.18 V.
        self.assertNotIn("voltage-sag", codes(stats))

    def test_falls_back_to_a_live_sample_when_there_is_no_reference(self):
        header = self._header(0)
        self.assertEqual(_detect_cells(header, fallback_volts=16.8), 4)

    def test_per_cell_voltage_and_the_sag_warning_follow_the_cell_count(self):
        session = Session(build_log(duration=1.0, seed=3))
        stats = session.stats()
        # The synthetic pack is 4S with vbatref 16.80 V.
        self.assertEqual(stats.cells, 4)
        self.assertAlmostEqual(stats.v_per_cell_min, stats.v_min / 4, places=6)
        self.assertNotIn("voltage-sag", codes(stats))

    def test_centivolt_conversion_is_the_default(self):
        session = Session(build_log(duration=0.2))
        self.assertAlmostEqual(_to_volts(1680, session.header, False), 16.8, places=3)

    def test_legacy_adc_conversion(self):
        session = Session(build_log(duration=0.2))
        # The Cleanflight formula: raw * 330 * scale / 4095 / 10, in millivolts.
        expected = 3000 * 330 * session.header.vbat_scale / 0xFFF / 10 / 1000.0
        self.assertAlmostEqual(_to_volts(3000, session.header, True), expected)

    def test_a_gyro_scale_of_exactly_one_is_accepted(self):
        """Regression: real Betaflight 4.x logs carry gyro_scale = 1.0.

        The firmware converts to deg/s before logging, so 1.0 is the correct
        value and not a broken one. Rejecting it as implausible fell back to a
        raw-count scale and reported every gyro figure 16.4 times too small.
        """
        log = HAND_BUILT_LOG.replace(
            b"H vbatref:1600\n", b"H vbatref:1600\nH gyro_scale:0x3f800000\n"
        )
        header = Session(log).header
        self.assertEqual(header.gyro_scale, 1.0)
        self.assertEqual(_gyro_scale(header), 1.0)

    def test_skipped_iterations_are_not_counted_as_dropped(self):
        """Regression: a 1/16 logging rate is not a 94% data loss."""
        log = HAND_BUILT_LOG.replace(b"H P interval:1/1\n", b"H P interval:16\n")
        stats = Session(log).stats()
        self.assertEqual(stats.main_frames, 3)
        self.assertEqual(stats.dropped_iterations, 0)
        self.assertEqual(stats.dropout_pct, 0.0)
        self.assertNotIn("logging-dropouts", codes(stats))

    def test_genuine_dropouts_are_still_caught(self):
        """The counterpart: frames really missing must still be reported."""
        session = Session(build_log(duration=2.0, seed=5, corrupt=20))
        stats = session.stats()
        self.assertGreater(stats.dropped_iterations, 0)
        self.assertIn("logging-dropouts", codes(stats))

    def test_gyro_scale_override(self):
        session = Session(build_log(duration=0.5, seed=3))
        default = compute_stats(session).gyro_peak[0]
        doubled = compute_stats(session, gyro_scale=2 / 16.4).gyro_peak[0]
        self.assertAlmostEqual(doubled, default * 2, places=3)

    def test_mode_flag_names(self):
        self.assertEqual(decode_mode_flags(0b11), ["ARM", "ANGLE"])
        self.assertEqual(decode_mode_flags(0), [])
        self.assertIn("FAILSAFE", decode_mode_flags(1 << 6))

    def test_distance_between_coordinates(self):
        # One degree of latitude is about 111 km anywhere on the globe.
        self.assertAlmostEqual(_haversine(51.4, -0.1, 52.4, -0.1) / 1000, 111, delta=1)
        self.assertEqual(_haversine(51.4, -0.1, 51.4, -0.1), 0.0)


if __name__ == "__main__":
    unittest.main()
