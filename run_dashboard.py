#!/usr/bin/env python3
"""
Script to run the Garmin Health Dashboard with day-of-week analysis.
"""

import sys
from pathlib import Path

# Support `python run_dashboard.py` from the repo without relying on editable layout details
_repo = Path(__file__).resolve().parent
for _p in (_repo / "src", _repo / "apps" / "dashboard"):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)

from garmin_dashboard.__main__ import main

if __name__ == "__main__":
    raise SystemExit(main())
