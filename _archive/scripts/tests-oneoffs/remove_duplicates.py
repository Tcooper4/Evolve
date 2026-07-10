#!/usr/bin/env python3
"""Remove duplicate optimization directories."""

import shutil
from pathlib import Path


def main():
    """Remove duplicate directories."""
    directories_to_remove = ["optimize", "optimizer", "optimizers"]

    for dir_name in directories_to_remove:
        dir_path = Path(dir_name)
        if dir_path.exists():
            try:
                shutil.rmtree(dir_path)
                print(f"Removed {dir_path}")
            except Exception as e:
                print(f"Error removing {dir_path}: {e}")
        else:
            print(f"{dir_path} does not exist")


if __name__ == "__main__":
    main()
