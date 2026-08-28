"""Command line interface: ``postflight``.

Subcommands are deliberately narrow -- look at a log, summarise a flight,
list what happened, or get the numbers out as CSV -- because the interesting
part of this project is the decoder, and a CLI that tries to be a plotting
suite would bury it.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import os
import sys

from . import __version__
from .decoder import Event, read_log
from .errors import BlackboxError
from .sparkline import bar, chart, supports_blocks
from .stats import decode_mode_flags

EXIT_OK = 0
EXIT_ERROR = 1


class Style:
    """Line-drawing characters, downgraded when the console cannot cope."""

    def __init__(self, unicode_ok):
        self.unicode = unicode_ok
        self.rule = "─" if unicode_ok else "-"
        self.arrow = "→" if unicode_ok else "->"
        self.bullet = "·" if unicode_ok else "*"
        self.warn = "⚠" if unicode_ok else "!"

    def heading(self, text, width=64):
        return "%s\n%s" % (text, self.rule * min(width, max(len(text), 24)))


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return EXIT_OK

    unicode_ok = supports_blocks() and not getattr(args, "ascii", False)
    style = Style(unicode_ok)

    try:
        return args.handler(args, style)
    except BlackboxError as error:
        print("error: %s" % error, file=sys.stderr)
        return EXIT_ERROR
    except BrokenPipeError:  # pragma: no cover - happens when piping to head
        # Must precede OSError, which it subclasses.
        return EXIT_OK
    except IsADirectoryError:
        print("error: %s is a directory, not a log file" % args.log, file=sys.stderr)
        return EXIT_ERROR
    except PermissionError:
        # Windows raises this rather than IsADirectoryError for a directory.
        if os.path.isdir(getattr(args, "log", "") or ""):
            message = "%s is a directory, not a log file" % args.log
        else:
            message = "cannot read %s: permission denied" % getattr(args, "log", "")
        print("error: %s" % message, file=sys.stderr)
        return EXIT_ERROR
    except OSError as error:
        # Anything else the filesystem throws: unreadable device, bad path,
        # a name too long. A traceback helps nobody here.
        print("error: %s" % error, file=sys.stderr)
        return EXIT_ERROR


def _build_parser():
    parser = argparse.ArgumentParser(
        prog="postflight",
        description="Decode and analyse Betaflight blackbox flight logs.",
    )
    parser.add_argument("--version", action="version", version="postflight " + __version__)
    subparsers = parser.add_subparsers(dest="command")

    def add(name, help_text, needs_log=True):
        sub = subparsers.add_parser(name, help=help_text, description=help_text)
        if needs_log:
            sub.add_argument("log", help="path to a .BBL / .BFL / .TXT blackbox log")
            sub.add_argument(
                "-s",
                "--session",
                type=int,
                default=None,
                help="only this flight, numbered from 1 (default: all of them)",
            )
        sub.add_argument(
            "--ascii", action="store_true", help="plain ASCII output, no block glyphs"
        )
        return sub

    info = add("info", "Show the header of each flight in a log.")
    info.add_argument("--fields", action="store_true", help="list every logged field")
    info.set_defaults(handler=cmd_info)

    stats = add("stats", "Summarise each flight: battery, motors, gyro, warnings.")
    stats.add_argument("--json", action="store_true", help="machine-readable output")
    stats.add_argument("--no-charts", action="store_true", help="skip the sparklines")
    stats.add_argument("--width", type=int, default=56, help="chart width in columns")
    stats.add_argument(
        "--vbat-adc",
        action="store_true",
        help="voltage is a raw ADC reading (Cleanflight-era logs)",
    )
    stats.set_defaults(handler=cmd_stats)

    events = add("events", "List events and mode changes in flight order.")
    events.set_defaults(handler=cmd_events)

    export = add("csv", "Export decoded frames as CSV.")
    export.add_argument("-o", "--out", help="output file (default: standard output)")
    export.add_argument(
        "-f",
        "--frame",
        default="main",
        choices=["main", "I", "P", "S", "G"],
        help="which frames to export (default: main, meaning I and P)",
    )
    export.add_argument(
        "--fields", help="comma-separated subset of field names to keep"
    )
    export.set_defaults(handler=cmd_csv)

    demo = add("demo", "Write a synthetic log to try the tool without a quad.", False)
    demo.add_argument("-o", "--out", default="demo.bbl", help="output path")
    demo.add_argument("--duration", type=float, default=12.0, help="flight seconds")
    demo.add_argument("--loop-hz", type=int, default=1000, help="PID loop rate")
    demo.add_argument("--seed", type=int, default=7, help="random seed")
    demo.add_argument("--gps", action="store_true", help="include GPS frames")
    demo.add_argument("--crash", action="store_true", help="end it in a crash")
    demo.add_argument("--rx-glitch", action="store_true", help="add an RX dropout")
    demo.add_argument(
        "--corrupt", type=int, default=0, help="corrupt this many bytes of the body"
    )
    demo.add_argument(
        "--sessions", type=int, default=1, help="how many flights to concatenate"
    )
    demo.set_defaults(handler=cmd_demo)
    return parser


def _sessions(args):
    """Read the log once and return (selected sessions, how many it holds)."""
    sessions = read_log(args.log)
    if not sessions:
        raise BlackboxError("%s contains no blackbox sessions" % args.log)
    wanted = getattr(args, "session", None)
    if wanted is None:
        return sessions, len(sessions)
    if not 1 <= wanted <= len(sessions):
        raise BlackboxError(
            "flight %d does not exist, the log has %d" % (wanted, len(sessions))
        )
    return [sessions[wanted - 1]], len(sessions)


def _title(session, total, style):
    parts = [session.header.craft_name or "(unnamed craft)", session.header.firmware]
    if session.header.board:
        parts.append(session.header.board)
    label = (" %s " % style.bullet).join(part for part in parts if part)
    return "flight %d of %d %s %s" % (session.index + 1, total, style.bullet, label)


def _has_main_frames(session):
    """Did this session record any flight data at all?

    A flight controller writes a header the moment it arms, so a file can hold
    plenty of sessions that were disarmed again before a single frame landed.
    Stops at the first one it finds, so this stays cheap on a real log.
    """
    return any(frame.kind in ("I", "P") for frame in session.frames())


# -- commands -------------------------------------------------------------
def cmd_info(args, style):
    sessions, total = _sessions(args)
    size = os.path.getsize(args.log)
    print(
        "%s  %s  %d flight%s"
        % (args.log, _bytes(size), total, "" if total == 1 else "s")
    )

    for session in sessions:
        header = session.header
        title = _title(session, total, style)
        if not _has_main_frames(session):
            title += " (no frames)"
        print()
        print(style.heading(title))
        _row("logged at", header.log_start_datetime or "(not recorded)")
        _row("firmware", "%s, built %s" % (header.firmware, header.firmware_date or "?"))
        _row("data version", header.data_version)
        _row("loop time", "%d us" % header.looptime if header.looptime else "?")
        interval = header.p_interval
        _row(
            "logging rate",
            "1 intra frame every %d, %d/%d inter frames"
            % (header.frame_interval_i, interval[0], interval[1]),
        )
        low, high = header.motor_output_range
        if low is not None:
            _row("motor output", "%d..%d" % (low, high))
        _row("battery ref", "%.2f V" % (header.vbat_ref / 100.0))
        cells = header.cell_voltage_limits
        if cells:
            _row("cell limits", "%.2f / %.2f / %.2f V" % tuple(v / 100.0 for v in cells))
        scale = header.gyro_scale
        if scale:
            _row("gyro scale", "%.6f deg/s per count" % scale)
        _row("size", "%s of frame data" % _bytes(len(session.data)))
        _row(
            "frame types",
            ", ".join(
                "%s(%d fields)" % (kind, len(frame))
                for kind, frame in sorted(session.header.frames.items())
            ),
        )
        if args.fields:
            for kind, frame in sorted(header.frames.items()):
                print("\n  %s frame fields:" % kind)
                for line in _wrap(frame.names, 68):
                    print("    " + line)
    return EXIT_OK


def cmd_stats(args, style):
    sessions, total_sessions = _sessions(args)
    reports = []

    for session in sessions:
        stats = session.stats(vbat_adc=args.vbat_adc)
        if args.json:
            payload = stats.as_dict()
            payload["flight"] = session.index + 1
            reports.append(payload)
            continue
        _print_report(session, stats, style, args, total_sessions)

    if args.json:
        json.dump(reports if len(reports) != 1 else reports[0], sys.stdout, indent=2)
        sys.stdout.write("\n")
    return EXIT_OK


def _print_report(session, stats, style, args, total_sessions):
    print()
    print(style.heading(_title(session, total_sessions, style)))

    minutes, seconds = divmod(stats.duration_s, 60)
    _row(
        "duration",
        "%dm %04.1fs" % (minutes, seconds) if minutes else "%.1f s" % seconds,
    )
    interval = session.header.p_interval
    _row(
        "loop rate",
        "%.0f Hz flight controller, %.0f Hz logged"
        % (stats.loop_hz, stats.loop_hz * interval[0] / interval[1]),
    )
    _row(
        "frames",
        ", ".join(
            "%d %s" % (count, kind)
            for kind, count in sorted(stats.frame_counts.items())
        ),
    )
    _row(
        "coverage",
        "%d of %d iterations logged (%.2f%% dropped)"
        % (stats.main_frames, stats.expected_iterations, stats.dropout_pct),
    )
    if stats.decode.corrupt_frames:
        _row(
            "corruption",
            "%d bad frames, %d bytes skipped, %d resyncs"
            % (
                stats.decode.corrupt_frames,
                stats.decode.corrupt_bytes,
                stats.decode.resyncs,
            ),
        )

    print()
    if stats.motor_mean:
        _row(
            "motor output",
            "mean %.0f%%  peak %.0f%%  %s"
            % (
                100 * stats.throttle_mean,
                100 * stats.throttle_max,
                bar(stats.throttle_mean, 16, style.unicode),
            ),
        )
        _row(
            "motors",
            "peak " + " / ".join("%.0f%%" % (100 * value) for value in stats.motor_max),
        )
    if stats.v_start is not None:
        per_cell = stats.v_per_cell_min
        _row(
            "battery",
            "%.2f V %s %.2f V  (%dS, min %.2f V/cell)"
            % (stats.v_start, style.arrow, stats.v_end, stats.cells, per_cell or 0),
        )
    if stats.amps_max:
        _row("current", "peak %.1f A, %.0f mAh used" % (stats.amps_max, stats.mah_used))
    if stats.rssi_min is not None:
        _row("receiver", "RSSI low point %d" % stats.rssi_min)
    if any(stats.gyro_peak):
        _row(
            "gyro",
            "peak %s deg/s"
            % " / ".join("%.0f" % value for value in stats.gyro_peak),
        )
        _row(
            "gyro rms",
            "%s deg/s (whole signal, not filtered noise)"
            % " / ".join("%.0f" % value for value in stats.gyro_rms),
        )
    if stats.acc_peak_g:
        _row("accelerometer", "peak %.1f g" % stats.acc_peak_g)
    if stats.gps_sats is not None:
        details = ["%d satellites" % stats.gps_sats]
        if stats.gps_speed_max_kmh:
            details.append("max %.0f km/h" % stats.gps_speed_max_kmh)
        if stats.gps_distance_max_m:
            details.append("%.0f m from home" % stats.gps_distance_max_m)
        _row("gps", ", ".join(details))

    if not args.no_charts and stats.series.get("throttle"):
        print()
        width = max(16, args.width)
        print(chart("motor out", stats.series["throttle"], width, unicode=style.unicode))
        if stats.series.get("voltage"):
            print(
                chart("voltage", stats.series["voltage"], width, "V", style.unicode)
            )
        if stats.series.get("current"):
            print(chart("current", stats.series["current"], width, "A", style.unicode))
        if stats.series.get("gyro"):
            print(chart("gyro", stats.series["gyro"], width, unicode=style.unicode))

    if stats.events:
        print("\nevents")
        for event in stats.events:
            print(("  " + _format_event(event, stats)).rstrip())

    if stats.anomalies:
        print("\nwarnings")
        for anomaly in stats.anomalies:
            print("  %s %s: %s" % (style.warn, anomaly["code"], anomaly["detail"]))
    else:
        print("\nno warnings")


def cmd_events(args, style):
    sessions, total_sessions = _sessions(args)
    for session in sessions:
        print()
        print(style.heading(_title(session, total_sessions, style)))
        start = None
        found = False
        for item in session.frames():
            if not isinstance(item, Event):
                if start is None and item.kind in ("I", "P"):
                    start = item.get("time", 0)
                continue
            found = True
            time_us = item.get("time", 0)
            relative = (time_us - (start or 0)) / 1_000_000
            detail = _event_detail(item)
            print(("  %8.3f s  %-12s %s" % (relative, item.name, detail)).rstrip())
        if not found:
            print("  (no events logged)")
    return EXIT_OK


def cmd_csv(args, style):
    sessions, _total = _sessions(args)
    if len(sessions) > 1:
        raise BlackboxError(
            "csv export needs a single flight, pass --session 1..%d" % len(sessions)
        )
    session = sessions[0]

    kinds = ("I", "P") if args.frame == "main" else (args.frame,)
    frame_def = session.header.frames.get("I" if args.frame == "main" else args.frame)
    if frame_def is None:
        raise BlackboxError("this log has no %s frames" % args.frame)

    names = frame_def.names
    if args.fields:
        wanted = [name.strip() for name in args.fields.split(",") if name.strip()]
        missing = [name for name in wanted if name not in frame_def.index]
        if missing:
            raise BlackboxError("no such field: %s" % ", ".join(missing))
        names = wanted

    # Either a file we own and must close, or stdout which we must not. On
    # Windows a text-mode stdout translates the writer's \r\n again, giving
    # \r\r\n and a malformed file, so ask for no translation on this stream.
    if args.out:
        handle = open(args.out, "w", newline="", encoding="utf-8")  # noqa: SIM115
    else:
        handle = sys.stdout
        reconfigure = getattr(handle, "reconfigure", None)
        if reconfigure is not None:
            with contextlib.suppress(ValueError, OSError):
                reconfigure(newline="")
    try:
        writer = csv.writer(handle)
        writer.writerow(names)
        rows = 0
        for frame in session.frames():
            if isinstance(frame, Event) or frame.kind not in kinds:
                continue
            writer.writerow([frame.get(name, "") for name in names])
            rows += 1
    finally:
        if args.out:
            handle.close()
    if args.out:
        print("wrote %d rows to %s" % (rows, args.out))
    return EXIT_OK


def cmd_demo(args, style):
    from .synth import build_log, build_multi_session_log

    options = {
        "duration": args.duration,
        "loop_hz": args.loop_hz,
        "seed": args.seed,
        "gps": args.gps,
        "corrupt": args.corrupt,
        "crash_at": args.duration * 0.78 if args.crash else None,
        "rx_glitch_at": args.duration * 0.4 if args.rx_glitch else None,
    }
    if args.sessions > 1:
        data = build_multi_session_log(args.sessions, **options)
    else:
        data = build_log(**options)

    with open(args.out, "wb") as handle:
        handle.write(data)
    print(
        "wrote %s (%s, %d flight%s) -- try: postflight stats %s"
        % (
            args.out,
            _bytes(len(data)),
            args.sessions,
            "" if args.sessions == 1 else "s",
            args.out,
        )
    )
    return EXIT_OK


# -- formatting helpers ---------------------------------------------------
def _row(label, value):
    print("  %-14s %s" % (label, value))


def _bytes(count):
    for unit in ("B", "KB", "MB", "GB"):
        if count < 1024 or unit == "GB":
            return "%.0f %s" % (count, unit) if unit == "B" else "%.1f %s" % (count, unit)
        count /= 1024.0
    return "%d B" % count


def _wrap(items, width):
    lines = []
    current = ""
    for item in items:
        piece = item if not current else ", " + item
        if len(current) + len(piece) > width:
            lines.append(current)
            current = item
        else:
            current += piece
    if current:
        lines.append(current)
    return lines


def _event_detail(event):
    if event.name == "disarm":
        return event.get("reasonName", "")
    if event.name == "flightMode":
        modes = decode_mode_flags(event.get("flags", 0))
        return ", ".join(modes) if modes else "none"
    if event.name == "loggingResume":
        return "iteration %d" % event.get("iteration", 0)
    if event.name == "inflightAdjustment":
        return "function %s = %s" % (event.get("function"), event.get("value"))
    return ""


def _format_event(entry, stats):
    start = stats.start_time_us or 0
    relative = (entry.get("time", 0) - start) / 1_000_000
    detail = entry.get("reason") or ", ".join(entry.get("modes", []))
    return "%8.3f s  %-12s %s" % (relative, entry["type"], detail)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
