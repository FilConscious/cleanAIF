from pathlib import Path

# Dynamically resolve the project root
current_script_path = Path(__file__).resolve()
PROJECT_ROOT = current_script_path.parents[2]  # Go two levels up to the project root
# New paths for logging and results
LOG_DIR = PROJECT_ROOT / "logs"
RESULTS_DIR = PROJECT_ROOT / "results"
# Create directories at the given paths, if they do not exist
LOG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
