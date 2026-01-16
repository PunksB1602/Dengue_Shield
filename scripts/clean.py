
from pathlib import Path
import shutil

print("Cleaning Python artifacts...")

# Remove __pycache__ directories
for p in Path(".").rglob("__pycache__"):
    shutil.rmtree(p, ignore_errors=True)
    print(f"Removed: {p}")

# Remove .pyc files
for p in Path(".").rglob("*.pyc"):
    p.unlink(missing_ok=True)

print("Done!")
