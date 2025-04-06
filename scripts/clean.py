#!/usr/bin/env python3
import os
from pathlib import Path
import shutil
import subprocess
import sys

import toml  # Added import

# ===== Project Configuration =====
# These lists define patterns for project-specific cleaning.
# Consider loading these from pyproject.toml ([tool.scripts.clean])
# or another configuration file for better project isolation.
ADDITIONAL_DIRS_TO_CLEAN = []  # e.g., ["logs", "temp_data"]
ADDITIONAL_FILES_TO_CLEAN = []  # e.g., ["*.log", "*.tmp"]
# ================================

# UV clean command configuration
UV_CLEAN_COMMAND = ["uv", "cache", "clean", "--all"]


def get_project_name(toml_file_path: str = "pyproject.toml") -> str | None:
    """
    Reads the pyproject.toml file and returns the project name.

    Args:
        toml_file_path (str): The path to the pyproject.toml file. Defaults to "pyproject.toml".

    Returns:
        str | None: The project name, or None if it cannot be found.
    """
    try:
        with open(toml_file_path, encoding="utf-8") as f:
            data = toml.load(f)
            # Navigate through potential structures [tool.poetry.name] or [project.name]
            if "project" in data and "name" in data["project"]:
                return data["project"]["name"]
            elif "tool" in data and "poetry" in data["tool"] and "name" in data["tool"]["poetry"]:
                return data["tool"]["poetry"]["name"]
            else:
                print(
                    "Error: Could not find project name in pyproject.toml under [project.name] or [tool.poetry.name]"
                )
                return None
    except FileNotFoundError:
        print(f"Error: {toml_file_path} not found.")
        return None
    except (KeyError, toml.TomlDecodeError) as e:
        print(f"Error reading {toml_file_path}: {e}")
        return None


def print_separator(message=""):
    """Print a separator line with an optional message."""
    try:
        width = shutil.get_terminal_size().columns - 10
    except OSError:  # Fallback if terminal size cannot be determined
        width = 70
    if message:
        print(f"\n{'-' * 10} {message} {'-' * max(0, width - len(message) - 12)}")
    else:
        print(f"\n{'-' * max(10, width)}")


def remove_directory(path: Path):
    """Remove a directory if it exists."""
    if path.is_dir():  # Ensure it's actually a directory
        try:
            print(f"Removing directory: {path}")
            shutil.rmtree(path)
            return True
        except OSError as e:  # Catch specific OS errors
            print(f"Error removing directory {path}: {e}")
            return False
    elif path.exists():
        print(f"Warning: Expected directory but found file: {path}. Skipping removal.")
        return False
    # If it doesn't exist, it's technically "removed" or clean.
    # print(f"Directory not found (already clean): {path}")
    return False  # Return False as no action was taken


def remove_file(path: Path):
    """Remove a file if it exists."""
    if path.is_file():  # Ensure it's actually a file
        try:
            print(f"Removing file: {path}")
            path.unlink()
            return True
        except OSError as e:  # Catch specific OS errors
            print(f"Error removing file {path}: {e}")
            return False
    elif path.exists():
        print(f"Warning: Expected file but found directory: {path}. Skipping removal.")
        return False
    # If it doesn't exist, it's technically "removed" or clean.
    # print(f"File not found (already clean): {path}")
    return False  # Return False as no action was taken


def find_and_remove_patterns(root_dir: Path, patterns: list[str], remove_func):
    """Find and remove files/directories matching given glob patterns."""
    removed_count = 0
    for pattern in patterns:
        try:
            matches = list(root_dir.glob(pattern))
            if matches:
                print(f"Found {len(matches)} items matching pattern: '{pattern}'")
                for item_path in matches:
                    if remove_func(item_path):
                        removed_count += 1
            # else:
            #     print(f"No items found matching pattern: '{pattern}'")
        except Exception as e:
            print(f"Error processing pattern '{pattern}': {e}")
    return removed_count


def clean_standard_artifacts(root_dir: Path):
    """Clean common Python build/cache artifacts."""
    total_removed = 0
    print_separator("Cleaning Standard Python Artifacts")

    # __pycache__ directories
    total_removed += find_and_remove_patterns(root_dir, ["**/__pycache__"], remove_directory)
    # .pyc files
    total_removed += find_and_remove_patterns(root_dir, ["**/*.pyc"], remove_file)
    # .pytest_cache directories
    total_removed += find_and_remove_patterns(root_dir, ["**/.pytest_cache"], remove_directory)
    # .mypy_cache directories
    total_removed += find_and_remove_patterns(root_dir, ["**/.mypy_cache"], remove_directory)
    # coverage files and directories
    total_removed += find_and_remove_patterns(root_dir, [".coverage", ".coverage.*"], remove_file)
    total_removed += find_and_remove_patterns(root_dir, ["htmlcov"], remove_directory)
    # .egg-info directories
    total_removed += find_and_remove_patterns(root_dir, ["**/*.egg-info"], remove_directory)
    # build/dist directories
    total_removed += find_and_remove_patterns(root_dir, ["build", "dist"], remove_directory)
    # PyInstaller .spec files
    total_removed += find_and_remove_patterns(root_dir, ["*.spec"], remove_file)

    print(f"Removed {total_removed} standard artifact items.")
    return total_removed


def run_command(command):
    """Run a command and return the result."""
    try:
        print(f"Executing: {' '.join(command)}")
        # Use encoding for cross-platform compatibility
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, encoding="utf-8"
        )
        if result.stdout and result.stdout.strip():
            print(result.stdout.strip())
        return result.stdout, None
    except subprocess.CalledProcessError as e:
        error_message = f"Command failed with exit code {e.returncode}"
        if e.stdout and e.stdout.strip():
            error_message += f"\nSTDOUT:\n{e.stdout.strip()}"
        if e.stderr and e.stderr.strip():
            error_message += f"\nSTDERR:\n{e.stderr.strip()}"
        return None, error_message
    except FileNotFoundError:
        return None, f"Error: Command not found - ensure '{command[0]}' is installed and in PATH."
    except Exception as e:
        return None, f"An unexpected error occurred: {e}"


def run_uv_clean():
    """Run UV cache clean command."""
    print_separator("Cleaning UV Cache")
    stdout, stderr = run_command(UV_CLEAN_COMMAND)

    if stderr:
        print(f"UV cache clean failed:\n{stderr}")
        return False

    print("UV cache cleaned successfully.")
    return True


def clean_additional_items(root_dir: Path):
    """Clean additional project-specific files and directories."""
    total_removed = 0
    if not ADDITIONAL_DIRS_TO_CLEAN and not ADDITIONAL_FILES_TO_CLEAN:
        print("No additional project-specific items configured for cleaning.")
        return 0

    print_separator("Cleaning Additional Project Items")
    if ADDITIONAL_DIRS_TO_CLEAN:
        print(f"Cleaning additional directories: {ADDITIONAL_DIRS_TO_CLEAN}")
        # Use find_and_remove_patterns for consistency
        total_removed += find_and_remove_patterns(
            root_dir, ADDITIONAL_DIRS_TO_CLEAN, remove_directory
        )

    if ADDITIONAL_FILES_TO_CLEAN:
        print(f"Cleaning additional files/patterns: {ADDITIONAL_FILES_TO_CLEAN}")
        # Use find_and_remove_patterns for consistency
        total_removed += find_and_remove_patterns(root_dir, ADDITIONAL_FILES_TO_CLEAN, remove_file)

    print(f"Removed {total_removed} additional project items.")
    return total_removed


def main():
    """Main clean function."""
    print("Generic Python Project Clean Script")
    print("===================================")

    # --- Determine Project Root ---
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    print(f"Working directory set to: {os.getcwd()}")
    root_dir = Path(".")  # Use relative path from the changed directory

    # --- Get Project Name (Optional for clean, but good practice) ---
    project_name = get_project_name()
    if project_name:
        print(f"Cleaning project: {project_name}")
    else:
        print("Warning: Could not determine project name from pyproject.toml.")
        # Decide if this should be a fatal error for clean script
        # return 1

    # --- Clean Standard Artifacts ---
    clean_standard_artifacts(root_dir)

    # --- Clean Additional Project-Specific Items ---
    clean_additional_items(root_dir)

    # --- Clean UV Cache ---
    run_uv_clean()

    print_separator()
    print("Cleanup completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
