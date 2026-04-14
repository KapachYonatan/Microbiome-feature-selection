from __future__ import annotations

from datetime import datetime


def log_checkpoint(message: str, section: bool = False) -> None:
    """Print a timestamped log checkpoint message.

    Input:
    - message: human-readable status text.
    - section: if True, prints section separators around the message.

    Output:
    - No return value. Writes logs to stdout.
    """

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if section:
        print(f"\n{'=' * 70}")
        print(f"[{timestamp}] {message}")
        print(f"{'=' * 70}\n")
    else:
        print(f"[{timestamp}] {message}")
