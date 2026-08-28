"""Header parsing: field definitions and the system configuration block."""

import unittest

from postflight.errors import CorruptLog
from postflight.header import parse_header

FULL_HEADER = (
    b"H Product:Blackbox flight data recorder by Nicholas Sherlock\n"
    b"H Data version:2\n"
    b"H I interval:32\n"
    b"H P interval:1/2\n"
    b"H Field I name:loopIteration,time,gyroADC[0],gyroADC[1]\n"
    b"H Field I signed:0,0,1,1\n"
    b"H Field I predictor:0,0,0,0\n"
    b"H Field I encoding:1,1,0,0\n"
    b"H Field P predictor:6,2,1,1\n"
    b"H Field P encoding:9,0,0,0\n"
    b"H Field S name:flightModeFlags,stateFlags\n"
    b"H Field S signed:0,0\n"
    b"H Field S predictor:0,0\n"
    b"H Field S encoding:1,1\n"
    b"H Firmware type:Cleanflight\n"
    b"H Firmware revision:Betaflight 4.4.0 (release) OMNIBUSF4\n"
    b"H Firmware date:Feb 24 2023 03:20:07\n"
    b"H Board information:BTFL OBF4\n"
    b"H Craft name:Toothpick\n"
    b"H looptime:125\n"
    b"H minthrottle:1070\n"
    b"H maxthrottle:2000\n"
    b"H motorOutput:48,2047\n"
    b"H gyro.scale:0x3d79c190\n"
    b"H acc_1G:2048\n"
    b"H vbat_scale:110\n"
    b"H vbatcellvoltage:330,350,430\n"
    b"H vbatref:1660\n"
    b"H currentSensor:0,400\n"
    b"H Log start datetime:2026-08-28T18:41:07.000+00:00\n"
    b"I\x00"
)


class HeaderTest(unittest.TestCase):
    def setUp(self):
        self.header = parse_header(FULL_HEADER)

    def test_text_headers(self):
        self.assertEqual(self.header.craft_name, "Toothpick")
        self.assertEqual(self.header.firmware, "Betaflight 4.4.0 (release) OMNIBUSF4")
        self.assertEqual(self.header.board, "BTFL OBF4")
        self.assertEqual(self.header.firmware_date, "Feb 24 2023 03:20:07")
        self.assertTrue(self.header.log_start_datetime.startswith("2026-08-28"))
        self.assertEqual(self.header.data_version, 2)

    def test_numeric_headers(self):
        self.assertEqual(self.header.looptime, 125)
        self.assertEqual(self.header.min_throttle, 1070)
        self.assertEqual(self.header.max_throttle, 2000)
        self.assertEqual(self.header.motor_output_range, (48, 2047))
        self.assertEqual(self.header.acc_1g, 2048)
        self.assertEqual(self.header.vbat_ref, 1660)
        self.assertEqual(self.header.vbat_scale, 110)
        self.assertEqual(self.header.cell_voltage_limits, (330, 350, 430))
        self.assertEqual(self.header.current_sensor, (0, 400))
        self.assertEqual(self.header.frame_interval_i, 32)
        self.assertEqual(self.header.p_interval, (1, 2))

    def test_gyro_scale_is_a_float_in_disguise(self):
        self.assertAlmostEqual(self.header.gyro_scale, 1 / 16.4, places=6)

    def test_frame_definitions(self):
        intra = self.header.frames["I"]
        self.assertEqual(intra.names, ["loopIteration", "time", "gyroADC[0]", "gyroADC[1]"])
        self.assertEqual([field.signed for field in intra.fields], [False, False, True, True])
        self.assertEqual([field.encoding for field in intra.fields], [1, 1, 0, 0])
        self.assertEqual(len(intra), 4)
        self.assertEqual(intra.index["gyroADC[1]"], 3)

        inter = self.header.frames["P"]
        self.assertEqual(inter.names, intra.names, "P frames inherit I frame names")
        self.assertEqual([field.predictor for field in inter.fields], [6, 2, 1, 1])
        self.assertEqual([field.encoding for field in inter.fields], [9, 0, 0, 0])

        self.assertEqual(self.header.frames["S"].names, ["flightModeFlags", "stateFlags"])
        self.assertNotIn("G", self.header.frames)

    def test_end_offset_points_at_the_first_frame(self):
        self.assertEqual(FULL_HEADER[self.header.end_offset], ord("I"))

    def test_has_field(self):
        self.assertTrue(self.header.has_field("gyroADC[0]"))
        self.assertFalse(self.header.has_field("gyroADC[2]"))
        self.assertTrue(self.header.has_field("stateFlags", "S"))
        self.assertFalse(self.header.has_field("stateFlags", "G"))

    def test_repr(self):
        self.assertIn("Toothpick", repr(self.header))


class HeaderFallbackTest(unittest.TestCase):
    def _header(self, *replacements):
        raw = FULL_HEADER
        for old, new in replacements:
            raw = raw.replace(old, new)
        return parse_header(raw)

    def test_throttle_falls_back_to_the_digital_motor_range(self):
        header = self._header((b"H minthrottle:1070\n", b""), (b"H maxthrottle:2000\n", b""))
        self.assertEqual(header.min_throttle, 48)
        self.assertEqual(header.max_throttle, 2047)

    def test_missing_motor_output(self):
        header = self._header((b"H motorOutput:48,2047\n", b""))
        self.assertEqual(header.motor_output_range, (None, None))
        self.assertEqual(header.min_throttle, 1070)

    def test_p_denom_instead_of_p_interval(self):
        header = self._header((b"H P interval:1/2\n", b"H P denom:4\n"))
        self.assertEqual(header.p_interval, (1, 4))

    def test_bare_p_interval_is_a_denominator(self):
        # Betaflight 4.2 writes "P interval:16" with no slash, meaning one
        # inter frame per sixteen loop iterations.
        header = self._header((b"H P interval:1/2\n", b"H P interval:16\n"))
        self.assertEqual(header.p_interval, (1, 16))

    def test_unparsable_p_interval_falls_back(self):
        header = self._header((b"H P interval:1/2\n", b"H P interval:sixteen\n"))
        self.assertEqual(header.p_interval, (1, 1))

    def test_no_interval_headers_at_all(self):
        header = self._header((b"H P interval:1/2\n", b""), (b"H I interval:32\n", b""))
        self.assertEqual(header.p_interval, (1, 1))
        self.assertEqual(header.frame_interval_i, 32)

    def test_zero_padded_numbers_are_accepted(self):
        """int(x, 0) rejects "010"; one such header must not condemn a log."""
        header = self._header((b"H looptime:125\n", b"H looptime:0125\n"))
        self.assertEqual(header.looptime, 125)

    def test_zero_padded_field_lists_are_accepted(self):
        header = self._header(
            (b"H Field I predictor:0,0,0,0\n", b"H Field I predictor:00,0,00,0\n")
        )
        self.assertEqual([f.predictor for f in header.frames["I"].fields], [0, 0, 0, 0])

    def test_hex_values_still_parse(self):
        header = self._header((b"H acc_1G:2048\n", b"H acc_1G:0x800\n"))
        self.assertEqual(header.acc_1g, 2048)
        self.assertAlmostEqual(header.gyro_scale, 1 / 16.4, places=6)

    def test_unparsable_values_do_not_kill_the_log(self):
        header = self._header((b"H vbatref:1660\n", b"H vbatref:not-a-number\n"))
        self.assertEqual(header.vbat_ref, 0)

    def test_bad_gyro_scale(self):
        header = self._header((b"H gyro.scale:0x3d79c190\n", b"H gyro.scale:wat\n"))
        self.assertIsNone(header.gyro_scale)

    def test_line_without_a_colon_is_ignored(self):
        header = self._header((b"H Craft name:Toothpick\n", b"H a stray line\n"))
        self.assertEqual(header.craft_name, "")

    def test_non_integer_field_list_is_fatal(self):
        with self.assertRaises(CorruptLog):
            self._header((b"H Field I signed:0,0,1,1\n", b"H Field I signed:0,0,x,1\n"))


if __name__ == "__main__":
    unittest.main()
