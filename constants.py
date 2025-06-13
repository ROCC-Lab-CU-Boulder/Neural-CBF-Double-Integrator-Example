"""Hold global variables for the session."""
import os

def _get_identifier() -> str:
    """Get the newest identifier for storing session information in the metrics directory."""
    # parse previous run directories
    base_dir = "./metrics"
    if "metrics" not in os.listdir("./"):
        os.mkdir("./metrics")
        print("Created directory './metrics' as it was not found")
    existing_runs = [
        d for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d)) and d.startswith("run-") and d[4:].isdigit()
    ]
    run_numbers = [int(d[4:]) for d in existing_runs]
    next_run_number = max(run_numbers, default=0) + 1
    return str(next_run_number)

identifier: str = _get_identifier()
