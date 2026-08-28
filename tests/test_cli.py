"""End-to-end tests for the postflight command line."""

import csv
import io
import json
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

from postflight import read_log
from postflight.cli import main


def run(*argv):
    """Run the CLI, returning (exit code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        code = main(list(argv))
    return code, out.getvalue(), err.getvalue()


class CliTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="postflight-test-")
        cls.log = os.path.join(cls.tmp, "flight.bbl")
        cls.multi = os.path.join(cls.tmp, "multi.bbl")
        run("demo", "-o", cls.log, "--duration", "2", "--gps", "--crash")
        run("demo", "-o", cls.multi, "--duration", "1", "--sessions", "2")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)


class DemoCommandTest(CliTestCase):
    def test_demo_writes_a_readable_log(self):
        path = os.path.join(self.tmp, "made-up.bbl")
        code, out, _ = run("demo", "-o", path, "--duration", "1")
        self.assertEqual(code, 0)
        self.assertTrue(os.path.getsize(path) > 1000)
        self.assertIn("postflight stats", out)

    def test_demo_can_make_several_flights(self):
        code, out, _ = run("info", self.multi)
        self.assertEqual(code, 0)
        self.assertIn("2 flights", out)
        self.assertIn("flight 2 of 2", out)


class InfoCommandTest(CliTestCase):
    def test_info_reports_the_header(self):
        code, out, _ = run("info", self.log)
        self.assertEqual(code, 0)
        self.assertIn("SynthQuad", out)
        self.assertIn("Betaflight 4.4.0", out)
        self.assertIn("motor output", out)
        self.assertIn("gyro scale", out)

    def test_sessions_without_flight_data_are_marked(self):
        """A flight controller writes a header on arm, frames only if it flies.

        Real files hold plenty of header-only sessions, and listing them
        alongside real flights with no distinction is misleading.
        """
        empty = os.path.join(self.tmp, "header-only.bbl")
        with open(self.log, "rb") as handle:
            flown = handle.read()
        # Cut at the parsed header boundary; searching the bytes for "\nI"
        # finds that pair inside the binary frame data instead.
        header_end = read_log(self.log)[0].header.end_offset
        with open(empty, "wb") as handle:
            handle.write(flown[:header_end] + b"E\xffEnd of log\x00" + flown)

        code, out, _ = run("info", empty)
        self.assertEqual(code, 0)
        lines = [line for line in out.splitlines() if line.startswith("flight ")]
        self.assertEqual(len(lines), 2)
        self.assertIn("(no frames)", lines[0])
        self.assertNotIn("(no frames)", lines[1])

    def test_info_can_list_every_field(self):
        code, out, _ = run("info", self.log, "--fields")
        self.assertEqual(code, 0)
        self.assertIn("loopIteration", out)
        self.assertIn("motor[3]", out)
        self.assertIn("GPS_coord[0]", out)


class StatsCommandTest(CliTestCase):
    def test_human_readable_report(self):
        code, out, _ = run("stats", self.log)
        self.assertEqual(code, 0)
        # "motor output" rather than "throttle": the figure is the mean motor
        # output, not the throttle stick, and the report now says so.
        for expected in ("duration", "loop rate", "motor output", "battery", "gyro"):
            self.assertIn(expected, out)
        self.assertNotIn("gyro noise", out)
        self.assertIn("warnings", out)
        self.assertIn("impact", out, "the crash demo should report an impact")

    def test_charts_can_be_switched_off(self):
        _, with_charts, _ = run("stats", self.log)
        _, without, _ = run("stats", self.log, "--no-charts")
        self.assertLess(len(without), len(with_charts))

    def test_json_output_is_machine_readable(self):
        code, out, _ = run("stats", self.log, "--json")
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertEqual(payload["craft"], "SynthQuad")
        self.assertEqual(payload["cells"], 4)
        self.assertEqual(payload["mainFrames"], 2000)
        self.assertTrue(payload["anomalies"])

    def test_json_output_for_several_flights_is_a_list(self):
        code, out, _ = run("stats", self.multi, "--json", "--no-charts")
        self.assertEqual(code, 0)
        payload = json.loads(out)
        self.assertIsInstance(payload, list)
        self.assertEqual(len(payload), 2)
        self.assertEqual([entry["flight"] for entry in payload], [1, 2])

    def test_selecting_one_flight(self):
        code, out, _ = run("stats", self.multi, "--session", "2", "--no-charts")
        self.assertEqual(code, 0)
        self.assertIn("flight 2 of 2", out)
        self.assertNotIn("flight 1 of 2", out)

    def test_selecting_a_flight_that_is_not_there(self):
        code, _, err = run("stats", self.multi, "--session", "9")
        self.assertEqual(code, 1)
        self.assertIn("does not exist", err)


class EventsCommandTest(CliTestCase):
    def test_events_are_listed_in_order(self):
        code, out, _ = run("events", self.log)
        self.assertEqual(code, 0)
        self.assertIn("flightMode", out)
        self.assertIn("disarm", out)
        self.assertIn("SWITCH", out)
        times = [
            float(line.split("s")[0]) for line in out.splitlines() if line.endswith("s")
        ]
        self.assertEqual(times, sorted(times))


class CsvCommandTest(CliTestCase):
    def test_csv_to_stdout(self):
        code, out, _ = run("csv", self.log, "--session", "1")
        self.assertEqual(code, 0)
        rows = list(csv.reader(io.StringIO(out)))
        self.assertEqual(rows[0][:2], ["loopIteration", "time"])
        self.assertEqual(len(rows), 2001)  # header plus every main frame
        self.assertEqual(rows[1][0], "0")

    def test_csv_to_a_file_with_selected_fields(self):
        path = os.path.join(self.tmp, "out.csv")
        code, out, _ = run(
            "csv", self.log, "--session", "1", "--fields", "time,motor[0]", "-o", path
        )
        self.assertEqual(code, 0)
        self.assertIn("wrote 2000 rows", out)
        with open(path, encoding="utf-8") as handle:
            rows = list(csv.reader(handle))
        self.assertEqual(rows[0], ["time", "motor[0]"])
        self.assertEqual(len(rows), 2001)

    def test_csv_of_gps_frames(self):
        code, out, _ = run("csv", self.log, "--session", "1", "--frame", "G")
        self.assertEqual(code, 0)
        rows = list(csv.reader(io.StringIO(out)))
        self.assertIn("GPS_numSat", rows[0])
        self.assertGreater(len(rows), 2)

    def test_csv_rejects_an_unknown_field(self):
        code, _, err = run("csv", self.log, "--session", "1", "--fields", "nope")
        self.assertEqual(code, 1)
        self.assertIn("no such field", err)

    def test_csv_needs_one_flight_at_a_time(self):
        code, _, err = run("csv", self.multi)
        self.assertEqual(code, 1)
        self.assertIn("single flight", err)


class CsvLineEndingTest(CliTestCase):
    """CSV must be well formed however it leaves the program.

    On Windows a text-mode stdout translates the csv module's \\r\\n a second
    time, producing \\r\\r\\n and a file that strict readers choke on. The
    -o path was already correct; stdout was not.
    """

    def _rows_are_well_formed(self, data):
        self.assertNotIn(b"\r\r\n", data)
        text = data.decode("utf-8")
        rows = list(csv.reader(io.StringIO(text)))
        self.assertTrue(all(row for row in rows), "a blank row crept in")
        return rows

    def test_file_output_line_endings(self):
        path = os.path.join(self.tmp, "endings-file.csv")
        run("csv", self.log, "--session", "1", "--fields", "time", "-o", path)
        with open(path, "rb") as handle:
            rows = self._rows_are_well_formed(handle.read())
        self.assertEqual(rows[0], ["time"])
        self.assertEqual(len(rows), 2001)

    def test_stdout_output_line_endings(self):
        # Route real bytes through a file handle rather than a StringIO, so
        # the newline translation this guards against can actually happen.
        path = os.path.join(self.tmp, "endings-stdout.csv")
        argv = ["csv", self.log, "--session", "1", "--fields", "time"]
        saved = sys.stdout
        try:
            with open(path, "w", encoding="utf-8") as handle:
                sys.stdout = handle
                code = main(argv)
        finally:
            sys.stdout = saved
        self.assertEqual(code, 0)
        with open(path, "rb") as handle:
            rows = self._rows_are_well_formed(handle.read())
        self.assertEqual(rows[0], ["time"])
        self.assertEqual(len(rows), 2001)

    def test_both_destinations_agree(self):
        path = os.path.join(self.tmp, "same.csv")
        _, piped, _ = run("csv", self.log, "--session", "1", "--fields", "time")
        run("csv", self.log, "--session", "1", "--fields", "time", "-o", path)
        with open(path, encoding="utf-8", newline="") as handle:
            self.assertEqual(handle.read().replace("\r\n", "\n"), piped.replace("\r\n", "\n"))


class ErrorHandlingTest(CliTestCase):
    def test_a_directory_is_a_clean_error_not_a_traceback(self):
        directory = os.path.join(self.tmp, "a-directory")
        os.makedirs(directory, exist_ok=True)
        for command in ("info", "stats", "events", "csv"):
            code, out, err = run(command, directory)
            self.assertEqual(code, 1, command)
            self.assertIn("directory", err.lower(), command)
            self.assertNotIn("Traceback", err, command)
            self.assertNotIn("Traceback", out, command)

    def test_missing_file(self):
        code, _, err = run("stats", os.path.join(self.tmp, "nope.bbl"))
        self.assertEqual(code, 1)
        self.assertTrue(err)

    def test_file_that_is_not_a_log(self):
        path = os.path.join(self.tmp, "notalog.bbl")
        with open(path, "wb") as handle:
            handle.write(b"just some bytes" * 100)
        code, _, err = run("info", path)
        self.assertEqual(code, 1)
        self.assertIn("no blackbox sessions", err)

    def test_no_arguments_prints_help(self):
        code, out, _ = run()
        self.assertEqual(code, 0)
        self.assertIn("usage", out)

    def test_version(self):
        with self.assertRaises(SystemExit) as caught:
            run("--version")
        self.assertEqual(caught.exception.code, 0)

    def test_ascii_mode_emits_no_wide_characters(self):
        code, out, _ = run("stats", self.log, "--ascii")
        self.assertEqual(code, 0)
        out.encode("ascii")  # raises if any glyph slipped through


if __name__ == "__main__":
    unittest.main()
