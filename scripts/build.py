#!/usr/bin/env python3
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys

import toml  # Added import

# ===== Project Configuration =====
# These variables might need further adaptation or configuration loading
# depending on project conventions (e.g., reading from pyproject.toml).
# ENTRY_POINT = "main.py" # Example: Consider reading from [project.scripts]
# HIDDEN_IMPORTS = [] # Example: Consider inferring or reading from config
# ================================

# Test command configuration (Consider making this configurable)
TEST_COMMAND = ["uv", "run", "pytest", "-v"]
# Build command configuration (Consider making this configurable)
BUILD_COMMAND = ["uv", "build"]


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


def run_command(command, cwd=None):
    """Run a command and return the result."""
    try:
        print(f"Executing: {' '.join(command)}")
        result = subprocess.run(
            command, capture_output=True, text=True, check=True, cwd=cwd, encoding="utf-8"
        )
        # Print stdout only if it's not empty
        if result.stdout and result.stdout.strip():
            print(result.stdout.strip())
        return result.stdout, None
    except subprocess.CalledProcessError as e:
        # Print stdout and stderr from the error if they exist
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


def run_tests():
    """Run all tests and return True if they pass."""
    print_separator("Running Tests")
    stdout, stderr = run_command(TEST_COMMAND)

    if stderr:
        print(f"Test execution failed:\n{stderr}")
        return False

    # Success message is printed within run_command if stdout exists
    print("Tests completed.")
    # Assume success if no stderr was produced (specific checks might be needed based on test runner output)
    return True


def create_build_directory():
    """Create or clean the build and dist directories."""
    build_dir = Path("build")
    dist_dir = Path("dist")

    print_separator("Preparing Build Directories")
    for directory in [build_dir, dist_dir]:
        if directory.exists():
            print(f"Cleaning {directory} directory...")
            try:
                shutil.rmtree(directory)
            except OSError as e:
                print(f"Warning: Could not remove {directory}: {e}. It might be in use.")
                # Optionally decide whether to continue or fail
        try:
            directory.mkdir(parents=True, exist_ok=True)
            print(f"Ensured {directory} directory exists.")
        except OSError as e:
            print(f"Error: Could not create directory {directory}: {e}")
            return None, None  # Indicate failure

    return build_dir, dist_dir


def build_wheel_package():
    """Build wheel package using UV build."""
    print_separator("Building Wheel Package")
    stdout, stderr = run_command(BUILD_COMMAND)

    if stderr:
        print(f"Wheel build failed:\n{stderr}")
        return False

    wheel_files = list(Path("dist").glob("*.whl"))
    if wheel_files:
        print(f"Wheel package created successfully: {[str(f) for f in wheel_files]}")
        return True
    else:
        # Check stdout for potential success messages even if no file is found immediately
        if stdout and "Built wheel" in stdout:
            print(
                "Wheel build reported success, but no .whl file found in dist/. Check build output."
            )
            # Decide if this is a failure or just a warning
            return False  # Treat as failure for safety
        print("Wheel build failed - no output file found and no success message in output.")
        return False


def build_platform_executable(project_name, dist_dir, entry_point=None, hidden_imports=None):
    """Build platform-specific executable using PyInstaller."""
    current_platform = platform.system()
    if current_platform not in ["Windows", "Darwin"]:  # Darwin is MacOS
        print(
            f"Platform {current_platform} not supported for direct executable compilation by this script."
        )
        print("Skipping executable build...")
        return True  # Not a failure, just skipping

    platform_name = "Windows" if current_platform == "Windows" else "MacOS"
    print_separator(f"Building {platform_name} Executable")

    # Ensure PyInstaller is installed in the environment UV uses
    print("Ensuring PyInstaller is installed...")
    stdout_install, stderr_install = run_command(["uv", "pip", "install", "pyinstaller"])
    if stderr_install:
        print(f"PyInstaller installation check/install failed:\n{stderr_install}")
        # Attempt to continue, PyInstaller might be globally available or in PATH
        # return False # Or uncomment to fail hard

    # Determine Entry Point
    if not entry_point:
        # Try to find a default entry point (e.g., src/project_name/main.py or project_name/main.py)
        possible_entry_points = [
            Path(f"src/{project_name}/{project_name}.py"),
            Path(f"src/{project_name}/main.py"),
            Path(f"{project_name}/{project_name}.py"),
            Path(f"{project_name}/main.py"),
            Path("main.py"),
        ]
        entry_point_path = next((p for p in possible_entry_points if p.exists()), None)
        if not entry_point_path:
            print("Error: Could not automatically determine entry point (e.g., main.py).")
            print(
                f"Please specify ENTRY_POINT in the script or ensure one of {possible_entry_points} exists."
            )
            return False
        entry_point = str(entry_point_path)
        print(f"Using automatically detected entry point: {entry_point}")

    # Determine Hidden Imports (Example: look for common patterns or read from config)
    if hidden_imports is None:
        hidden_imports = []
        # Basic heuristic: add top-level packages/modules within the project source
        src_dir = (
            Path(f"src/{project_name}")
            if Path(f"src/{project_name}").is_dir()
            else Path(project_name)
        )
        if src_dir.is_dir():
            potential_imports = [f.stem for f in src_dir.glob("*.py") if f.stem != "__init__"]
            potential_imports.extend([
                d.name for d in src_dir.glob("*") if d.is_dir() and (d / "__init__.py").exists()
            ])
            # Filter common stdlib/generic names if needed
            hidden_imports.extend([f"{project_name}.{mod}" for mod in potential_imports])
            if hidden_imports:
                print(
                    f"Automatically added hidden imports based on source structure: {hidden_imports}"
                )
        # Add more sophisticated detection or configuration reading here if needed

    pyinstaller_command = [
        "uv",
        "run",
        "pyinstaller",
        "--onefile",
        "--name",
        project_name,
        "--clean",  # Clean PyInstaller cache and temporary files
        # Add current directory to path; adjust if source is elsewhere (e.g., src/)
        "--paths",
        ".",
        # Consider adding '--paths src' if your code is in a src layout
    ]

    # Add all hidden imports
    for hidden_import in hidden_imports:
        pyinstaller_command.extend(["--hidden-import", hidden_import])

    # Add entry point
    pyinstaller_command.append(entry_point)

    print(f"Running PyInstaller for {platform_name}...")
    stdout_build, stderr_build = run_command(pyinstaller_command)

    if stderr_build:
        print(f"{platform_name} executable build failed:\n{stderr_build}")
        return False

    # Verify executable creation
    exe_suffix = ".exe" if current_platform == "Windows" else ""
    exe_path = Path(f"dist/{project_name}{exe_suffix}")

    if exe_path.exists():
        print(f"{platform_name} executable created successfully at: {exe_path}")
        return True
    else:
        print(f"{platform_name} executable build failed - output file not found at {exe_path}.")
        if stdout_build:
            print(f"PyInstaller Output:\n{stdout_build}")  # Show output for debugging
        return False


def verify_build(dist_dir):
    """Verify the build outputs exist."""
    print_separator("Verifying Build Outputs")

    if not dist_dir or not dist_dir.exists():
        print(f"Error: Distribution directory '{dist_dir}' not found.")
        return False

    outputs = list(dist_dir.glob("*"))
    if not outputs:
        print("No build outputs found in dist/ directory!")
        return False  # Treat as failure

    print("Build outputs found in dist/:")
    for output in outputs:
        print(f"  - {output.name}")

    # Add more specific verification if needed (e.g., check file sizes, types)
    return True


def main():
    """Main build function."""
    print("Generic Python Project Build Script")
    print("===================================")

    # --- Determine Project Root ---
    # Assume the script is run from the root or scripts/ directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    print(f"Working directory set to: {os.getcwd()}")

    # --- Get Project Name ---
    project_name = get_project_name()
    if not project_name:
        print("Build failed: Could not determine project name from pyproject.toml.")
        return 1
    print(f"Detected Project Name: {project_name}")

    # --- Run Tests ---
    if not run_tests():
        print("Build failed: Tests failed.")
        return 1

    # --- Create Build/Dist Directories ---
    build_dir, dist_dir = create_build_directory()
    if not build_dir or not dist_dir:
        print("Build failed: Could not create build/dist directories.")
        return 1

    # --- Build Python Package (Wheel) ---
    if not build_wheel_package():
        print("Build failed: Package build failed.")
        return 1

    # --- Build Platform-Specific Executable ---
    # Pass dynamic project_name. Entry point and hidden imports might need adjustment.
    # Example: Read ENTRY_POINT and HIDDEN_IMPORTS from pyproject.toml if defined there
    # entry_point_config = get_config_from_toml("tool.build.entry_point")
    # hidden_imports_config = get_config_from_toml("tool.build.hidden_imports")
    if not build_platform_executable(
        project_name, dist_dir
    ):  # Add entry_point=..., hidden_imports=... if configured
        print("Build failed: Executable build process failed.")
        return 1

    # --- Verify Build ---
    if not verify_build(dist_dir):
        print("Build failed: Verification of build outputs failed.")
        return 1

    print_separator()
    print("Build completed successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
