import tomllib
from pathlib import Path
import sys

def list_project_scripts():
    """Reads pyproject.toml from the project root and lists the defined scripts."""
    # Assume the script is run from the project root where pyproject.toml exists
    pyproject_path = Path("pyproject.toml")

    if not pyproject_path.is_file():
        print(f"Error: {pyproject_path} not found in the current directory.", file=sys.stderr)
        print("Please run this script from the project root.", file=sys.stderr)
        sys.exit(1)

    try:
        with open(pyproject_path, "rb") as f:
            data = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        print(f"Error parsing {pyproject_path}: {e}", file=sys.stderr)
        sys.exit(1)
    except OSError as e:
        print(f"Error reading {pyproject_path}: {e}", file=sys.stderr)
        sys.exit(1)

    # Navigate through the structure [project][scripts]
    scripts = data.get("project", {}).get("scripts")

    print("Defined project scripts ([project.scripts]):")
    if not scripts:
        print("  (None)")
    else:
        # Find the maximum length of script names for alignment
        max_len = max(len(name) for name in scripts) if scripts else 0
        for name, target in sorted(scripts.items()): # Sort alphabetically for consistent output
            print(f"  {name:<{max_len}} : {target}")

def main():
    """Entry point for the script."""
    list_project_scripts()

if __name__ == "__main__":
    main()
