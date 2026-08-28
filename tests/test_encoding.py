"""Round-trip and golden-byte tests for the variable-length codecs."""

import random
import unittest

from postflight.encoding import (
    Reader,
    Writer,
    sign_extend,
    zigzag_decode,
    zigzag_encode,
)
from postflight.errors import CorruptLog, EndOfStream

INT32_MIN = -(2**31)
INT32_MAX = 2**31 - 1


class SignExtendTest(unittest.TestCase):
    def test_two_bit(self):
        self.assertEqual([sign_extend(v, 2) for v in range(4)], [0, 1, -2, -1])

    def test_four_bit(self):
        self.assertEqual(sign_extend(0x0F, 4), -1)
        self.assertEqual(sign_extend(0x08, 4), -8)
        self.assertEqual(sign_extend(0x07, 4), 7)

    def test_wider(self):
        self.assertEqual(sign_extend(0xFFFF, 16), -1)
        self.assertEqual(sign_extend(0x800000, 24), -(2**23))
        self.assertEqual(sign_extend(0xFFFFFFFF, 32), -1)


class ZigZagTest(unittest.TestCase):
    def test_known_pairs(self):
        for value, encoded in [(0, 0), (-1, 1), (1, 2), (-2, 3), (2, 4)]:
            self.assertEqual(zigzag_encode(value), encoded)
            self.assertEqual(zigzag_decode(encoded), value)

    def test_round_trip_extremes(self):
        for value in (INT32_MIN, INT32_MAX, -1, 0, 1, 12345, -12345):
            self.assertEqual(zigzag_decode(zigzag_encode(value)), value)


class VariableByteTest(unittest.TestCase):
    def test_golden_bytes(self):
        # 300 == 0b100101100 -> low seven bits with the continuation flag, then 2
        writer = Writer()
        writer.write_unsigned_vb(300)
        self.assertEqual(writer.bytes(), b"\xac\x02")

    def test_single_byte_below_128(self):
        writer = Writer()
        writer.write_unsigned_vb(127)
        self.assertEqual(writer.bytes(), b"\x7f")

    def test_unsigned_round_trip(self):
        values = [0, 1, 127, 128, 255, 16383, 16384, 2**28, 0xFFFFFFFF]
        writer = Writer()
        for value in values:
            writer.write_unsigned_vb(value)
        reader = Reader(writer.bytes())
        self.assertEqual([reader.read_unsigned_vb() for _ in values], values)
        self.assertTrue(reader.eof())

    def test_signed_round_trip(self):
        values = [0, -1, 1, -64, 64, INT32_MIN, INT32_MAX, -99999, 99999]
        writer = Writer()
        for value in values:
            writer.write_signed_vb(value)
        reader = Reader(writer.bytes())
        self.assertEqual([reader.read_signed_vb() for _ in values], values)

    def test_truncated_stream_raises(self):
        reader = Reader(b"\x80")  # continuation flag set, nothing follows
        with self.assertRaises(EndOfStream):
            reader.read_unsigned_vb()

    def test_overlong_varint_raises(self):
        reader = Reader(b"\x80" * 6)
        with self.assertRaises(CorruptLog):
            reader.read_unsigned_vb()


class Neg14BitTest(unittest.TestCase):
    def test_round_trip(self):
        for value in (0, -1, -100, -8191, 1, 8191):
            writer = Writer()
            writer.write_neg_14bit(value)
            self.assertEqual(Reader(writer.bytes()).read_neg_14bit(), value)


class Tag8_8SVBTest(unittest.TestCase):
    def test_single_field_skips_the_header_byte(self):
        writer = Writer()
        writer.write_tag8_8svb([-3])
        self.assertEqual(len(writer.bytes()), 1)
        self.assertEqual(Reader(writer.bytes()).read_tag8_8svb(1), [-3])

    def test_all_zero_costs_one_byte(self):
        writer = Writer()
        writer.write_tag8_8svb([0] * 8)
        self.assertEqual(writer.bytes(), b"\x00")
        self.assertEqual(Reader(writer.bytes()).read_tag8_8svb(8), [0] * 8)

    def test_sparse_group(self):
        values = [0, 5, 0, 0, -7, 0, 0, 3]
        writer = Writer()
        writer.write_tag8_8svb(values)
        self.assertEqual(Reader(writer.bytes()).read_tag8_8svb(8), values)

    def test_random_round_trip(self):
        rng = random.Random(1)
        for _ in range(200):
            count = rng.randint(2, 8)
            values = [rng.choice([0, rng.randint(-5000, 5000)]) for _ in range(count)]
            writer = Writer()
            writer.write_tag8_8svb(values)
            self.assertEqual(Reader(writer.bytes()).read_tag8_8svb(count), values)


class Tag2_3S32Test(unittest.TestCase):
    def test_selector_widths_are_chosen_tightly(self):
        cases = [
            ([1, -2, 0], 1),  # three 2-bit fields fit in the lead byte
            ([7, -8, 3], 2),  # 4-bit fields
            ([31, -32, 5], 3),  # 6-bit fields
            ([100, -100, 0], 4),  # lead byte plus three 8-bit fields
            ([30000, 1, 1], 5),  # one 16-bit field, two 8-bit
        ]
        for values, expected_len in cases:
            writer = Writer()
            writer.write_tag2_3s32(values)
            self.assertEqual(len(writer.bytes()), expected_len, values)
            self.assertEqual(Reader(writer.bytes()).read_tag2_3s32(), values)

    def test_wide_fields(self):
        values = [INT32_MIN, 2**20, -(2**20)]
        writer = Writer()
        writer.write_tag2_3s32(values)
        self.assertEqual(Reader(writer.bytes()).read_tag2_3s32(), values)

    def test_random_round_trip(self):
        rng = random.Random(2)
        for _ in range(500):
            bound = rng.choice([1, 8, 32, 128, 32768, 2**23, 2**31])
            values = [rng.randrange(-bound, bound) for _ in range(3)]
            writer = Writer()
            writer.write_tag2_3s32(values)
            reader = Reader(writer.bytes())
            self.assertEqual(reader.read_tag2_3s32(), values)
            self.assertTrue(reader.eof(), "decoder consumed the wrong byte count")

    def test_tag2_3svariable_is_not_implemented(self):
        """Encoding 10 has no reader, on purpose.

        It is declared in Betaflight's flightLogFieldEncoding_e but no field in
        blackbox.c uses it and blackbox-tools does not implement it, so there
        is no reference for its byte layout. An earlier version of this module
        aliased it to TAG2_3S32 on an unverified assumption; the decoder now
        rejects such a log instead of guessing.
        """
        self.assertFalse(hasattr(Reader(b""), "read_tag2_3svariable"))
        self.assertFalse(hasattr(Writer(), "write_tag2_3svariable"))


class Tag8_4S16Test(unittest.TestCase):
    def test_all_zero_costs_one_byte(self):
        writer = Writer()
        writer.write_tag8_4s16([0, 0, 0, 0])
        self.assertEqual(writer.bytes(), b"\x00")
        self.assertEqual(Reader(writer.bytes()).read_tag8_4s16(), [0, 0, 0, 0])

    def test_nibble_straddling_widths(self):
        # a 4-bit field first leaves the stream half a byte out of step, which
        # is the case the reference decoder handles with its nibble buffer
        for values in (
            [1, 100, 1000, 0],
            [-1, -100, -1000, 7],
            [7, 0, -32000, -8],
            [3, 3, 3, 3],
            [5, -20000, 6, 30000],
        ):
            writer = Writer()
            writer.write_tag8_4s16(values)
            self.assertEqual(Reader(writer.bytes()).read_tag8_4s16(), values, values)

    def test_random_round_trip(self):
        rng = random.Random(3)
        for _ in range(500):
            values = [
                rng.choice([0, rng.randrange(-8, 8), rng.randrange(-32768, 32768)])
                for _ in range(4)
            ]
            writer = Writer()
            writer.write_tag8_4s16(values)
            self.assertEqual(Reader(writer.bytes()).read_tag8_4s16(), values, values)


class Tag8_4S16VersionTest(unittest.TestCase):
    """Golden bytes for both layouts, worked out from the reference decoder.

    These expectations come from reading ``streamReadTag8_4S16_v1`` and
    ``streamReadTag8_4S16_v2`` in Betaflight's blackbox-tools, not from this
    package's own Writer, so they are independent of it. The two versions
    disagree on the same input, which is exactly why the log's Data version
    header has to select between them.
    """

    #: selector 0x55 = four 4-bit fields, then two payload bytes.
    NIBBLE_BYTES = bytes([0x55, 0x12, 0x34])

    def test_v2_takes_one_nibble_at_a_time_high_first(self):
        self.assertEqual(Reader(self.NIBBLE_BYTES).read_tag8_4s16(), [1, 2, 3, 4])

    def test_v1_takes_a_whole_byte_as_a_pair_low_nibble_first(self):
        self.assertEqual(Reader(self.NIBBLE_BYTES).read_tag8_4s16_v1(), [2, 1, 4, 3])

    def test_the_two_versions_disagree(self):
        self.assertNotEqual(
            Reader(self.NIBBLE_BYTES).read_tag8_4s16(),
            Reader(self.NIBBLE_BYTES).read_tag8_4s16_v1(),
        )

    def test_16_bit_fields_swap_endianness_between_versions(self):
        # selector 0x03 = field 0 is 16-bit, the rest are zero.
        data = bytes([0x03, 0x34, 0x12])
        self.assertEqual(Reader(data).read_tag8_4s16()[0], 0x3412)  # v2 big-endian
        self.assertEqual(Reader(data).read_tag8_4s16_v1()[0], 0x1234)  # v1 little

    def test_v1_zero_and_8_bit_fields(self):
        # The selector is read two bits at a time from the bottom, so
        # 0b00_10_00_10 describes fields 0..3 as 8-bit, zero, 8-bit, zero.
        data = bytes([0b00100010, 0xFF, 0x7F])
        self.assertEqual(Reader(data).read_tag8_4s16_v1(), [-1, 0, 127, 0])

    def test_v1_all_zero_costs_one_byte(self):
        self.assertEqual(Reader(b"\x00").read_tag8_4s16_v1(), [0, 0, 0, 0])


class ReaderTest(unittest.TestCase):
    def test_position_and_remaining(self):
        reader = Reader(b"\x01\x02\x03")
        self.assertEqual(reader.remaining, 3)
        self.assertEqual(reader.peek_byte(), 1)
        self.assertEqual(reader.pos, 0)
        self.assertEqual(reader.read_byte(), 1)
        self.assertEqual(reader.remaining, 2)
        self.assertFalse(reader.eof())

    def test_read_past_end(self):
        reader = Reader(b"")
        with self.assertRaises(EndOfStream):
            reader.read_byte()
        with self.assertRaises(EndOfStream):
            reader.peek_byte()


if __name__ == "__main__":
    unittest.main()
