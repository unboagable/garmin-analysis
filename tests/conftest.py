import os
import sys

# Ensure project root and src are on sys.path so `garmin_analysis` can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Use a non-interactive backend for matplotlib in tests
try:
    import matplotlib

    matplotlib.use("Agg")
except Exception:
    # If matplotlib is not installed for some environments, ignore
    pass


