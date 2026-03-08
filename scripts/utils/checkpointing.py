"""
Generic JSON checkpoint save / load utilities.
"""

import os
import json


def load_checkpoint(checkpoint_dir, key):
    """
    Load a checkpoint JSON file.

    Args:
        checkpoint_dir: Directory containing checkpoint files.
        key:            Filename stem (without .json).  The file
                        ``{checkpoint_dir}/{key}.json`` is read.

    Returns:
        Parsed dict, or None if the file does not exist or is unreadable.
    """
    filepath = os.path.join(checkpoint_dir, f"{key}.json")
    if not os.path.exists(filepath):
        return None
    try:
        with open(filepath, "r") as f:
            return json.load(f)
    except Exception:
        return None


def save_checkpoint(checkpoint_dir, key, data):
    """
    Serialise *data* as JSON to ``{checkpoint_dir}/{key}.json``.

    Args:
        checkpoint_dir: Target directory (created if absent).
        key:            Filename stem (without .json).
        data:           JSON-serialisable object.
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    filepath = os.path.join(checkpoint_dir, f"{key}.json")
    with open(filepath, "w") as f:
        json.dump(data, f)
