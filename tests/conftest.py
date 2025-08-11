import os
import sys

# Ensure project root is on sys.path so `src` can be imported
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Use a non-interactive backend for matplotlib in tests
try:
    import matplotlib

    matplotlib.use("Agg")
except Exception:
    # If matplotlib is not installed for some environments, ignore
    pass


