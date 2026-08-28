"""Terminal chart helpers, including the legacy-console fallback."""

import unittest

from postflight.sparkline import (
    ASCII_RAMP,
    BLOCKS,
    bar,
    chart,
    resample,
    sparkline,
    supports_blocks,
)


class FakeStream:
    def __init__(self, encoding):
        self.encoding = encoding


class SupportsBlocksTest(unittest.TestCase):
    def test_utf8_console(self):
        self.assertTrue(supports_blocks(FakeStream("utf-8")))

    def test_legacy_windows_code_page(self):
        self.assertFalse(supports_blocks(FakeStream("cp1252")))

    def test_stream_without_an_encoding(self):
        self.assertFalse(supports_blocks(FakeStream(None)))

    def test_nonsense_encoding_name(self):
        self.assertFalse(supports_blocks(FakeStream("not-a-real-codec")))


class ResampleTest(unittest.TestCase):
    def test_short_input_is_returned_whole(self):
        self.assertEqual(resample([1, 2, 3], 10), [1, 2, 3])

    def test_empty_input(self):
        self.assertEqual(resample([], 10), [])

    def test_buckets_are_averaged(self):
        self.assertEqual(resample([0, 2, 4, 6], 2), [1, 5])

    def test_output_length_matches_the_requested_width(self):
        self.assertEqual(len(resample(list(range(1000)), 40)), 40)


class SparklineTest(unittest.TestCase):
    def test_rising_series_ends_at_the_top_of_the_ramp(self):
        line = sparkline(list(range(50)), width=10, unicode=True)
        self.assertEqual(len(line), 10)
        self.assertEqual(line[-1], BLOCKS[-1])
        self.assertEqual(line[0], BLOCKS[0])

    def test_flat_series_uses_the_lowest_glyph(self):
        self.assertEqual(sparkline([5] * 8, width=8, unicode=True), BLOCKS[0] * 8)

    def test_ascii_fallback_uses_the_ascii_ramp(self):
        line = sparkline(list(range(20)), width=8, unicode=False)
        self.assertTrue(set(line) <= set(ASCII_RAMP))

    def test_explicit_bounds(self):
        # With the range forced wide, a small series should stay near the floor.
        line = sparkline([1, 2, 3], width=3, unicode=True, low=0, high=100)
        self.assertEqual(line, BLOCKS[0] * 3)

    def test_empty_series(self):
        self.assertEqual(sparkline([], width=10), "")


class BarTest(unittest.TestCase):
    def test_full_and_empty(self):
        self.assertEqual(bar(1.0, 10, unicode=False), "#" * 10)
        self.assertEqual(bar(0.0, 10, unicode=False), "." * 10)

    def test_half(self):
        self.assertEqual(bar(0.5, 10, unicode=False), "#####.....")

    def test_out_of_range_values_are_clamped(self):
        self.assertEqual(bar(4.2, 4, unicode=False), "####")
        self.assertEqual(bar(-1.0, 4, unicode=False), "....")


class ChartTest(unittest.TestCase):
    def test_label_and_range_are_shown(self):
        line = chart("voltage", [15.0, 16.8], width=8, unit="V", unicode=False)
        self.assertTrue(line.startswith("voltage"))
        self.assertIn("15.0V", line)
        self.assertIn("16.8V", line)

    def test_no_data(self):
        self.assertIn("no data", chart("gyro", [], width=8))


if __name__ == "__main__":
    unittest.main()
