# The Betaflight blackbox format

A walkthrough of the binary format `postflight` decodes, and of how this
implementation handles it. The [README](README.md) gives the short version;
this is the detail behind it.

Everything here is checked against the Betaflight firmware sources
(`blackbox.c`, `blackbox_fielddefs.h`) and the reference decoder in
blackbox-tools (`parser.c`, `decoders.c`) rather than inferred from data.

## File and session structure

A `.BBL` file off a flight controller usually holds **several sessions**, one
per arm/disarm cycle, simply concatenated. Each begins with the marker
`H Product:Blackbox`, which is how they are split apart.

A session has two parts back to back: an ASCII header, then binary frames.

## The header

Newline-terminated `H key:value` lines, plain ASCII, readable in any text
editor. Besides the craft name, firmware version and board, the header carries
the **field definitions** — for each frame type, the name, signedness,
predictor and encoding of every column:

```
H Field I name:loopIteration,time,axisP[0],motor[0],motor[1]
H Field I signed:0,0,1,0,0
H Field I predictor:0,0,0,4,5
H Field I encoding:1,1,0,1,0
H Field P predictor:6,2,1,3,3
H Field P encoding:9,0,0,0,0
```

Inter frames reuse the intra frame's column names and signs; only their
predictors and encodings differ, which is why only two `Field P` lines appear.

Without these definitions the binary payload cannot be decoded at all, which
is why a malformed header is fatal here while a malformed frame is not.

The header also carries the values needed to turn raw counts into physical
units — `vbatref`, `vbatcellvoltage`, `motorOutput`, `acc_1G`, `gyro_scale`,
`looptime`, and the `I interval` / `P interval` logging rates.

Two header quirks worth knowing, both found on real logs:

- `P interval` appears both as `1/16` and as a bare `16`. Both mean one inter
  frame per sixteen loop iterations.
- `gyro_scale` is the hex bit pattern of a float. Betaflight 4.x writes
  `0x3f800000`, which is exactly 1.0, because the firmware converts the gyro
  to degrees per second before logging it.

## Frame types

Each frame begins with a single type byte:

| Type | Meaning |
| --- | --- |
| `I` | Intra frame — every field stored absolutely. Written at the interval named by `I interval` (256 in the real logs tested) and used as a resynchronisation point |
| `P` | Inter frame — the common case, stored as a difference from a prediction |
| `S` | Slow frame — flight mode flags, failsafe phase, receiver state |
| `G` | GPS fix |
| `H` | GPS home position |
| `E` | Event — arming, mode change, disarm reason, end of log |

There is no length field and no checksum on any frame. A frame ends where its
fields end, which is what makes recovery from damage non-trivial.

## Prediction

This is where the compression comes from. Almost nothing is stored as its own
value: the firmware subtracts a guess and stores only the error, which is
usually small enough to fit in a byte or two.

| ID | Predictor | Guess |
| --- | --- | --- |
| 0 | `ZERO` | nothing — the value is absolute |
| 1 | `PREVIOUS` | the same field in the last frame |
| 2 | `STRAIGHT_LINE` | twice the previous value minus the one before it, for values changing at a steady rate such as the timestamp |
| 3 | `AVERAGE_2` | the mean of the last two, which smooths noisy sensors |
| 4 | `MINTHROTTLE` | the throttle floor from the header |
| 5 | `MOTOR_0` | motor 0 of the same frame — the other motors track it closely |
| 6 | `INCREMENT` | the previous value plus one, plus any intentionally skipped frames |
| 7 | `HOME_COORD` | the GPS home position for that axis |
| 8 | `1500` | the literal 1500, the centre of an RC channel |
| 9 | `VBATREF` | the battery reference voltage from the header |
| 10 | `LAST_MAIN_FRAME_TIME` | the timestamp of the last main frame |
| 11 | `MINMOTOR` | the low end of the `motorOutput` range |

Decoding means re-adding those predictions using exactly the arithmetic the
firmware used. Two details matter and are both wrong by default in Python:

- **32-bit wraparound.** The flight controller computed these in 32-bit
  registers, so every result is masked back to 32 bits and re-signed according
  to the field's declared signedness.
- **Truncating division.** `AVERAGE_2` halves a sum with C integer division,
  which truncates toward zero. Python's `//` floors, which differs for negative
  sums.

`INCREMENT` also has to know the logging rate: at `P interval` 1/16 the
firmware deliberately skips fifteen iterations out of every sixteen, so the
loop counter advances by sixteen per logged frame, not one.

## Timestamps and rollover

`time` is a 32-bit microsecond counter taken from the flight controller's
`micros()`, so it wraps every 71.6 minutes of uptime — not of flight, which
means a log written after a long bench session can cross the boundary.

The rule, matching `flightLogDetectAndApplyTimestampRollover` in the reference
decoder: a raw value that moved **backwards** but by only a **plausible**
amount in modulo-32-bit terms is a wrap, and 2³² is added to an accumulator.
Anything else that moves backwards is damage. Decoded timestamps are therefore
monotonic across a wrap, while a genuine backwards jump is still rejected.

## Encodings

The header names an encoding per field. IDs match Betaflight's
`flightLogFieldEncoding_e`; the gaps at 2, 4 and 5 are encodings removed from
the firmware years ago.

### `UNSIGNED_VB` (1) — variable-length integer

Seven bits of value per byte, least significant group first, with the high bit
set to mean "another byte follows". Values 0–127 cost one byte, up to 16383
cost two, and so on. Capped at five bytes, which is all a 32-bit value needs.
Since most stored values are small residuals, nearly everything in a real log
is a single byte.

```
E8 07  ->  0xE8 has the high bit set: keep 1101000 = 104, continue
           0x07 has it clear:         7 << 7      = 896, stop
                                                    = 1000
```

### `SIGNED_VB` (0) — zig-zag then varint

A varint cannot store −5 compactly: in two's complement its high bits are all
set, so it would take the full five bytes. Zig-zag interleaves the signs so
both stay small — 0, −1, 1, −2, 2 become 0, 1, 2, 3, 4 — and the result is
then written as a varint.

Encode is `(v << 1) ^ (v >> 31)`; decode is `(u >> 1) ^ -(u & 1)`, where the
second term is 0 or −1 and XOR with −1 flips every bit.

### `NEG_14BIT` (3)

An unsigned varint whose sign-extended low 14 bits are negated. Used for
battery voltage, which the firmware writes as `(vbatref - vbatLatest) & 0x3FFF`
— always a small positive drop from the reference.

### `TAG2_3S32` (7) — three values behind a width selector

A lead byte whose top two bits choose the layout of three signed values:

| Selector | Layout |
| --- | --- |
| 0 | three 2-bit fields, all inside the lead byte |
| 1 | three 4-bit fields |
| 2 | three 6-bit fields, one per byte |
| 3 | each field independently 8, 16, 24 or 32 bits, named by two bits each of the lead byte |

### `TAG8_8SVB` (6) — a bitmap of what is non-zero

A header byte with one bit per field, least significant bit for field 0, then
a signed varint for each field whose bit is set. An axis sitting at zero costs
one bit. The group runs until the encoding changes, up to eight fields. As a
special case, a group of exactly one field skips the bitmap entirely.

### `TAG8_4S16` (8) — four values, nibble packed

Four values behind a selector byte carrying two bits each: zero, 4-bit, 8-bit
or 16-bit. Because a 4-bit field leaves the stream half a byte out of step, a
value can begin mid-byte and the reader has to carry the leftover nibble
forward.

**This encoding has two layouts, and the header's `Data version` selects
between them.** They are genuinely different, and reading one with the other's
rules produces plausible but wrong numbers rather than an error:

| | Data version 1 | Data version 2 |
| --- | --- | --- |
| 4-bit fields | one byte carries two of them, low nibble first | one nibble at a time, high nibble first |
| 16-bit fields | little-endian | big-endian |

The same three bytes `55 12 34` decode to `[2, 1, 4, 3]` under version 1 and
`[1, 2, 3, 4]` under version 2. Both layouts are implemented, and that pair of
golden byte strings is a test.

### `NULL` (9)

Nothing is read; the value comes entirely from its predictor. Used for
`loopIteration` in inter frames, which is always the previous value plus the
frame interval.

### `TAG2_3SVARIABLE` (10) — not implemented, on purpose

Declared in Betaflight's enum, but no field in `blackbox.c` uses it and
blackbox-tools does not implement it either, so no reference exists for its
byte layout. A log declaring it is refused with a clear error rather than
decoded by guesswork. No real log tested uses it.

## Events

Betaflight emits exactly six event types, all decoded here: sync beep, flight
mode change, disarm (with reason), in-flight adjustment, logging resume, and
end of log.

The end-of-log event is followed by the literal string `"End of log"` and a NUL
byte, eleven bytes in total. Verifying that literal matters: without it, any
stray `0x45 0xFF` pair in damaged data would be read as a clean end of the
session.

Anything outside those six is treated as corruption. The reference decoder
assumes an unrecognised event carries no payload and keeps reading, which can
desynchronise the stream; resynchronising on a verified intra frame instead
costs data but cannot silently misread it.

## Corruption and resynchronisation

These logs are written to dataflash on a vehicle that gets crashed, and the
write path drops bytes when its buffer overruns. With no length fields and no
checksums, a bad byte can desynchronise everything after it.

Every main frame is therefore sanity-checked before it is accepted: time and
loop iteration must both move forward, and by no more than a plausible amount
— ten seconds and 5000 iterations, the same bounds the reference decoder uses.
The iteration bound has to clear the largest legitimate gap, which is one
intra-frame interval times the inter-frame denominator: 256 × 16 = 4096 on the
real logs tested.

When a frame fails, the decoder scans forward for the next intra frame,
decodes it, and accepts it only if it passes the same checks. Recovery costs at
most one intra-frame interval of data.

What this cannot do is detect damage that leaves a frame *syntactically*
valid — a flipped bit inside a varint is still a well-formed varint. That
limitation belongs to the format rather than to any decoder; see
[Known limitations](README.md#known-limitations) in the README.

## Where each piece lives

```
postflight/
  encoding.py   variable-length integer codecs, both directions
  header.py     the ASCII header block and field definitions
  decoder.py    the frame loop: predictors, validation, resynchronisation
  stats.py      one streaming pass for flight statistics and warnings
  sparkline.py  terminal charts, with an ASCII fallback for legacy consoles
  synth.py      a flight model and encoder, for the tests and `postflight demo`
  cli.py        the postflight command
```
