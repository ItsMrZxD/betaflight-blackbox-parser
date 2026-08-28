"""Variable-length integer codecs used by the Betaflight blackbox format.

Blackbox logs are written by a flight controller with a few kilobytes of RAM
onto slow dataflash, so every field is packed with the narrowest encoding that
fits its value. The log header names an encoding per field (see
:mod:`postflight.header`); this module implements them.

Both directions live here: :class:`Reader` decodes real logs, :class:`Writer`
is used by :mod:`postflight.synth` to generate logs for the test suite and for
``postflight demo``. Keeping them side by side makes the round-trip tests in
``tests/test_encoding.py`` a meaningful check of both.

Encoding IDs match Betaflight's FlightLogFieldEncoding enum; the gaps at 2, 4
and 5 are encodings that were removed from the firmware years ago.
"""

from __future__ import annotations

import struct

from .errors import CorruptLog, EndOfStream

SIGNED_VB = 0
UNSIGNED_VB = 1
NEG_14BIT = 3
TAG8_8SVB = 6
TAG2_3S32 = 7
TAG8_4S16 = 8
NULL = 9
TAG2_3SVARIABLE = 10

#: How many fields a single encoded group swallows. Every encoding not listed
#: here covers exactly one field.
GROUP_SIZE = {TAG8_8SVB: 8, TAG2_3S32: 3, TAG2_3SVARIABLE: 3, TAG8_4S16: 4}

ENCODING_NAMES = {
    SIGNED_VB: "SIGNED_VB",
    UNSIGNED_VB: "UNSIGNED_VB",
    NEG_14BIT: "NEG_14BIT",
    TAG8_8SVB: "TAG8_8SVB",
    TAG2_3S32: "TAG2_3S32",
    TAG8_4S16: "TAG8_4S16",
    NULL: "NULL",
    TAG2_3SVARIABLE: "TAG2_3SVARIABLE",
}


def sign_extend(value, bits):
    """Interpret the low `bits` of `value` as a two's complement integer."""
    sign_bit = 1 << (bits - 1)
    return (value & (sign_bit - 1)) - (value & sign_bit)


def zigzag_encode(value):
    """Map a signed int onto an unsigned one, keeping small magnitudes small."""
    return ((value << 1) ^ -1) & 0xFFFFFFFF if value < 0 else (value << 1) & 0xFFFFFFFF


def zigzag_decode(value):
    """Inverse of :func:`zigzag_encode`."""
    return (value >> 1) ^ -(value & 1)


class Reader:
    """A cursor over a bytes buffer with the blackbox codecs attached."""

    def __init__(self, data, pos=0):
        self.data = data
        self.pos = pos

    def __len__(self):
        return len(self.data)

    @property
    def remaining(self):
        return len(self.data) - self.pos

    def eof(self):
        return self.pos >= len(self.data)

    def read_byte(self):
        if self.pos >= len(self.data):
            raise EndOfStream("read past end of log at offset %d" % self.pos)
        byte = self.data[self.pos]
        self.pos += 1
        return byte

    def peek_byte(self):
        if self.pos >= len(self.data):
            raise EndOfStream("peek past end of log at offset %d" % self.pos)
        return self.data[self.pos]

    def read_unsigned_vb(self):
        """Little-endian base-128 varint; a set high bit means 'more to come'."""
        result = 0
        for shift in (0, 7, 14, 21, 28):
            byte = self.read_byte()
            result |= (byte & 0x7F) << shift
            if byte < 0x80:
                return result & 0xFFFFFFFF
        raise CorruptLog("varint longer than 5 bytes at offset %d" % self.pos)

    def read_signed_vb(self):
        return zigzag_decode(self.read_unsigned_vb())

    def read_neg_14bit(self):
        return -sign_extend(self.read_unsigned_vb(), 14)

    def read_tag8_8svb(self, count):
        """Up to 8 signed values behind a bitmap of which ones are non-zero."""
        if count == 1:
            return [self.read_signed_vb()]
        header = self.read_byte()
        return [self.read_signed_vb() if header & (1 << i) else 0 for i in range(count)]

    def read_tag2_3s32(self):
        """Three signed values, width chosen by the top 2 bits of a lead byte."""
        lead = self.read_byte()
        selector = lead >> 6
        if selector == 0:  # three 2-bit fields, all inside the lead byte
            return [
                sign_extend((lead >> 4) & 0x03, 2),
                sign_extend((lead >> 2) & 0x03, 2),
                sign_extend(lead & 0x03, 2),
            ]
        if selector == 1:  # three 4-bit fields
            first = sign_extend(lead & 0x0F, 4)
            packed = self.read_byte()
            return [first, sign_extend(packed >> 4, 4), sign_extend(packed & 0x0F, 4)]
        if selector == 2:  # three 6-bit fields, one per byte
            return [
                sign_extend(lead & 0x3F, 6),
                sign_extend(self.read_byte() & 0x3F, 6),
                sign_extend(self.read_byte() & 0x3F, 6),
            ]
        # selector 3: each field is 8, 16, 24 or 32 bits, named by 2 bits of lead
        values = []
        for _ in range(3):
            width = lead & 0x03
            if width == 0:
                values.append(sign_extend(self.read_byte(), 8))
            elif width == 1:
                values.append(sign_extend(self._read_le(2), 16))
            elif width == 2:
                values.append(sign_extend(self._read_le(3), 24))
            else:
                values.append(sign_extend(self._read_le(4), 32))
            lead >>= 2
        return values

    # There is deliberately no TAG2_3SVARIABLE reader. Encoding 10 is declared
    # in Betaflight's flightLogFieldEncoding_e but no field in blackbox.c ever
    # uses it, and blackbox-tools does not implement it either, so no ground
    # truth for its byte layout exists. The decoder rejects it rather than
    # guessing; see decoder._decode_fields.

    def read_tag8_4s16_v1(self):
        """The Data version 1 layout of TAG8_4S16.

        Older than v2 and genuinely different, so the log's ``Data version``
        header decides which one to use. Here a 4-bit field consumes a whole
        byte and carries *two* values -- the low nibble first, then the high
        nibble -- and 16-bit fields are little-endian. v2 packs 4-bit fields
        one nibble at a time and stores 16-bit fields big-endian, so reading a
        v1 log with v2 rules yields plausible but wrong numbers.
        """
        selector = self.read_byte()
        values = [0, 0, 0, 0]
        i = 0
        while i < 4:
            width = selector & 0x03
            if width == 0:
                values[i] = 0
            elif width == 1:  # one byte holding two 4-bit fields
                combined = self.read_byte()
                values[i] = sign_extend(combined & 0x0F, 4)
                if i + 1 < 4:
                    i += 1
                    selector >>= 2
                    values[i] = sign_extend(combined >> 4, 4)
            elif width == 2:
                values[i] = sign_extend(self.read_byte(), 8)
            else:  # 16 bits, little-endian in this version
                values[i] = sign_extend(self._read_le(2), 16)
            selector >>= 2
            i += 1
        return values

    def read_tag8_4s16(self):
        """Four signed values nibble-packed behind a 4x2-bit size selector.

        This is the Data version 2 layout, verified byte-for-byte against
        ``streamReadTag8_4S16_v2`` in Betaflight's blackbox-tools.
        """
        selector = self.read_byte()
        values = [0, 0, 0, 0]
        buffer = 0
        half = False  # True while the low nibble of `buffer` is still unread
        for i in range(4):
            width = selector & 0x03
            if width == 0:  # field is zero, costs no payload
                values[i] = 0
            elif width == 1:  # 4 bits
                if not half:
                    buffer = self.read_byte()
                    values[i] = sign_extend(buffer >> 4, 4)
                    half = True
                else:
                    values[i] = sign_extend(buffer & 0x0F, 4)
                    half = False
            elif width == 2:  # 8 bits, possibly straddling a nibble boundary
                if not half:
                    values[i] = sign_extend(self.read_byte(), 8)
                else:
                    byte, buffer = self._read_straddled(buffer)
                    values[i] = sign_extend(byte, 8)
            else:  # 16 bits, big-endian
                if not half:
                    values[i] = sign_extend(self._read_be(2), 16)
                else:
                    high, buffer = self._read_straddled(buffer)
                    low, buffer = self._read_straddled(buffer)
                    values[i] = sign_extend((high << 8) | low, 16)
            selector >>= 2
        return values

    def _read_straddled(self, buffer):
        """Pull one byte spanning the low nibble of `buffer` and a fresh byte."""
        byte = (buffer << 4) & 0xFF
        buffer = self.read_byte()
        return byte | (buffer >> 4), buffer

    def _read_le(self, size):
        value = 0
        for shift in range(0, size * 8, 8):
            value |= self.read_byte() << shift
        return value

    def _read_be(self, size):
        value = 0
        for _ in range(size):
            value = (value << 8) | self.read_byte()
        return value


class Writer:
    """Mirror image of :class:`Reader`, used to synthesise logs."""

    def __init__(self):
        self.buf = bytearray()

    def __len__(self):
        return len(self.buf)

    def bytes(self):
        return bytes(self.buf)

    def write_byte(self, value):
        self.buf.append(value & 0xFF)

    def write_raw(self, data):
        self.buf.extend(data)

    def write_unsigned_vb(self, value):
        value &= 0xFFFFFFFF
        while value > 0x7F:
            self.buf.append((value & 0x7F) | 0x80)
            value >>= 7
        self.buf.append(value)

    def write_signed_vb(self, value):
        self.write_unsigned_vb(zigzag_encode(value))

    def write_neg_14bit(self, value):
        self.write_unsigned_vb(-value & 0x3FFF)

    def write_tag8_8svb(self, values):
        if len(values) == 1:
            self.write_signed_vb(values[0])
            return
        header = 0
        for i, value in enumerate(values):
            if value != 0:
                header |= 1 << i
        self.write_byte(header)
        for value in values:
            if value != 0:
                self.write_signed_vb(value)

    def write_tag2_3s32(self, values):
        if all(-2 <= v <= 1 for v in values):
            self.write_byte(
                ((values[0] & 0x03) << 4)
                | ((values[1] & 0x03) << 2)
                | (values[2] & 0x03)
            )
        elif all(-8 <= v <= 7 for v in values):
            self.write_byte(0x40 | (values[0] & 0x0F))
            self.write_byte(((values[1] & 0x0F) << 4) | (values[2] & 0x0F))
        elif all(-32 <= v <= 31 for v in values):
            self.write_byte(0x80 | (values[0] & 0x3F))
            self.write_byte(values[1] & 0x3F)
            self.write_byte(values[2] & 0x3F)
        else:
            selector = 0xC0
            payload = bytearray()
            for i, value in enumerate(values):
                if -0x80 <= value <= 0x7F:
                    width, packed = 0, struct.pack("<b", value)
                elif -0x8000 <= value <= 0x7FFF:
                    width, packed = 1, struct.pack("<h", value)
                elif -0x800000 <= value <= 0x7FFFFF:
                    width, packed = 2, (value & 0xFFFFFF).to_bytes(3, "little")
                else:
                    width, packed = 3, struct.pack("<i", value)
                selector |= width << (i * 2)
                payload += packed
            self.write_byte(selector)
            self.write_raw(payload)

    def write_tag8_4s16(self, values):
        selector = 0
        for i, value in enumerate(values):
            if value == 0:
                width = 0
            elif -8 <= value <= 7:
                width = 1
            elif -0x80 <= value <= 0x7F:
                width = 2
            else:
                width = 3
            selector |= width << (i * 2)
        self.write_byte(selector)

        nibbles = []  # payload as a flat list of 4-bit chunks
        probe = selector
        for value in values:
            width = probe & 0x03
            probe >>= 2
            if width == 1:
                nibbles.append(value & 0x0F)
            elif width == 2:
                nibbles.extend(((value >> 4) & 0x0F, value & 0x0F))
            elif width == 3:
                nibbles.extend(
                    (
                        (value >> 12) & 0x0F,
                        (value >> 8) & 0x0F,
                        (value >> 4) & 0x0F,
                        value & 0x0F,
                    )
                )
        if len(nibbles) % 2:
            nibbles.append(0)  # pad the last byte; the reader never looks at it
        for high, low in zip(nibbles[0::2], nibbles[1::2]):
            self.write_byte((high << 4) | low)
