"""
Command-line entry point.

Usage:
    python -m src.cli --entry "some log text"          # log a single day
    python -m src.cli --demo data/sample_entries.txt    # replay a synthetic
                                                          multi-day log, one
                                                          agent turn per day
    python -m src.cli --report                          # print accumulated
                                                          flagged patterns
"""

import argparse
import logging
import sys
from datetime import date

from .agent import process_day, weekly_report


def _configure_logging(verbose: bool) -> None:
    """Route the agent's diagnostic logs to the console.

    Default (INFO) shows the agent's actions — tool calls, web searches — plus
    any warnings/errors. `--verbose` (DEBUG) adds the low-level per-call cache
    and stop-reason diagnostics. The two-space prefix keeps trace lines nested
    visually under the day banners printed by run_demo.
    """
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="  %(message)s",
    )


def run_demo(path: str) -> None:
    """Replay a synthetic multi-day log, running one agent turn per day."""
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()

    # Each day in the sample file is separated by a line starting with "DAY:"
    days = []
    current_day, current_text = None, []
    for line in raw.splitlines():
        if line.startswith("DAY:"):
            if current_day:
                days.append((current_day, "\n".join(current_text).strip()))
            current_day = line.split("DAY:", 1)[1].strip()
            current_text = []
        else:
            current_text.append(line)
    if current_day:
        days.append((current_day, "\n".join(current_text).strip()))

    for day, text in days:
        print(f"\n{'=' * 60}\nDAY {day}\n{'=' * 60}")
        print(f"Log: {text}\n")
        summary = process_day(text, day=day, ask_user=False)
        print(f"Agent: {summary}")

    print(f"\n{'=' * 60}\nWEEKLY REPORT\n{'=' * 60}")
    print(weekly_report())


def main():
    """Parse CLI args and dispatch to report / demo / single-entry logging."""
    parser = argparse.ArgumentParser(description="Symptom-Trigger Pattern Agent")
    parser.add_argument("--entry", help="Log a single day's entry as free text.")
    parser.add_argument("--date", help="Date to use when logging a single day's entry.")
    parser.add_argument("--demo", help="Path to a synthetic multi-day log to replay.")
    parser.add_argument("--report", action="store_true", help="Print the weekly report.")
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show low-level per-call diagnostics (cache hits, stop reasons).",
    )
    args = parser.parse_args()

    _configure_logging(args.verbose)

    if args.report:
        print(weekly_report())
    elif args.demo:
        run_demo(args.demo)
    elif args.entry:
        day = args.date if args.date else date.today().isoformat()
        print(process_day(args.entry, day=day))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
