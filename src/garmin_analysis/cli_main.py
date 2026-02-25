#!/usr/bin/env python3
"""
Main Garmin CLI with subcommands.

Usage:
  garmin init     — Bootstrap: check DBs, create folders, validate schema
"""

import argparse
import sys


def main():
    parser = argparse.ArgumentParser(
        prog="garmin",
        description="Garmin Analysis CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # garmin init
    init_p = subparsers.add_parser("init", help="Bootstrap: check DBs, create folders, validate schema")
    init_p.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    init_p.set_defaults(func=_run_init)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        return 0

    return args.func(args)


def _run_init(args):
    from garmin_analysis.cli_init import run_init
    return run_init(verbose=args.verbose)


if __name__ == "__main__":
    sys.exit(main())
