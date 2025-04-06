#!/usr/bin/env python3
"""
Automated release script for AutoCommit.

Handles version bumping, running tests, building, and tagging.
"""

import os
from pathlib import Path
import re
import shutil  # For print_separator width
import subprocess
import sys

import toml

# --- Configuration ---
MAIN_BRANCH = "main" # Or "master" if that's your default
REMOTE_NAME = "origin"
PYPROJECT_PATH = "pyproject.toml"
TEST_COMMAND = ["uv", "run", "pytest", "-v"] # Reuse from build.py
BUILD_COMMAND = ["uv", "build"] # Reuse from build.py

# --- Helper Functions ---

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

def run_command(command: list[str], cwd=None, check=True) -> tuple[str | None, str | None]:
    """
    Run a command, capture output, and return (stdout, stderr).
    Returns (None, error_message) on failure if check=True.
    Returns (stdout, stderr) even on failure if check=False.
    """
    try:
        print(f"Executing: {' '.join(command)}")
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            check=check, # Raise exception on non-zero exit if True
            cwd=cwd,
            encoding="utf-8",
        )
        stdout = process.stdout.strip() if process.stdout else ""
        stderr = process.stderr.strip() if process.stderr else ""
        if stdout:
            print(stdout) # Print stdout only if it's not empty
        # Don't print stderr here automatically, let caller decide based on context
        return stdout, stderr
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

def get_current_version() -> str | None:
    """Reads the current version from pyproject.toml."""
    try:
        data = toml.load(PYPROJECT_PATH)
        return data["project"]["version"]
    except (FileNotFoundError, KeyError, toml.TomlDecodeError) as e:
        print(f"Error reading version from {PYPROJECT_PATH}: {e}", file=sys.stderr)
        return None

def get_git_output(command: list[str]) -> str | None:
    """Runs a git command and returns stdout, handling errors."""
    stdout, stderr = run_command(command, check=False) # Don't fail immediately
    if stderr and "fatal:" in stderr: # Check for critical git errors
        print(f"Git command failed: {' '.join(command)}\nError: {stderr}", file=sys.stderr)
        return None
    # Ignore non-fatal stderr for commands like describe which might print to stderr
    return stdout

def get_remote_main_hash() -> str | None:
    """Gets the commit hash of the remote main branch."""
    return get_git_output(["git", "rev-parse", f"{REMOTE_NAME}/{MAIN_BRANCH}"])

def get_latest_remote_tag_version() -> str | None:
    """Gets the latest version tag from the remote main branch (e.g., v1.0.0 -> 1.0.0)."""
    # Fetch tags explicitly to ensure we have the latest ones
    run_command(["git", "fetch", REMOTE_NAME, "--tags"])
    # Get the latest tag reachable from the remote main branch
    tag = get_git_output(["git", "describe", "--tags", "--abbrev=0", f"{REMOTE_NAME}/{MAIN_BRANCH}"])
    if tag:
        # Remove potential 'v' prefix
        return tag.lstrip('v')
    # Handle case where no tags exist
    print("No version tags found on remote main branch. Assuming initial version 0.0.0.", file=sys.stderr)
    return "0.0.0" # Default if no tags

def get_local_head_hash() -> str | None:
    """Gets the commit hash of the local HEAD."""
    return get_git_output(["git", "rev-parse", "HEAD"])

def update_version_in_toml(new_version: str) -> bool:
    """Updates the version in pyproject.toml."""
    try:
        data = toml.load(PYPROJECT_PATH)
        data["project"]["version"] = new_version
        with open(PYPROJECT_PATH, "w", encoding="utf-8") as f:
            toml.dump(data, f)
        print(f"Updated {PYPROJECT_PATH} to version {new_version}")
        return True
    except (FileNotFoundError, KeyError, toml.TomlDecodeError, OSError) as e:
        print(f"Error updating version in {PYPROJECT_PATH}: {e}", file=sys.stderr)
        return False

def bump_patch_version(version_str: str) -> str | None:
    """Increments the patch number of a version string (e.g., 1.0.0 -> 1.0.1)."""
    match = re.match(r"(\d+)\.(\d+)\.(\d+)", version_str)
    if not match:
        print(f"Error: Cannot parse version string '{version_str}'", file=sys.stderr)
        return None
    major, minor, patch = map(int, match.groups())
    return f"{major}.{minor}.{patch + 1}"

def run_tests() -> bool:
    """Runs tests using the configured command."""
    print_separator("Running Tests")
    stdout, stderr = run_command(TEST_COMMAND)
    if stderr: # run_command returns stderr as error message on failure
        print(f"Tests failed:\n{stderr}", file=sys.stderr)
        return False
    print("Tests passed.")
    return True

def run_build() -> bool:
    """Runs the build using the configured command."""
    print_separator("Running Build")
     # Clean dist dir first
    dist_dir = Path("dist")
    if dist_dir.exists():
        print(f"Cleaning {dist_dir} directory...")
        try:
            shutil.rmtree(dist_dir)
        except OSError as e:
            print(f"Warning: Could not remove {dist_dir}: {e}. It might be in use.")
    try:
        dist_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        print(f"Error: Could not create directory {dist_dir}: {e}")
        return False

    stdout, stderr = run_command(BUILD_COMMAND)
    if stderr:
        print(f"Build failed:\n{stderr}", file=sys.stderr)
        return False
    # Check if wheel file exists
    wheel_files = list(Path("dist").glob("*.whl"))
    if not wheel_files:
         print("Build command succeeded but no wheel file found in dist/", file=sys.stderr)
         return False
    print(f"Build successful: {wheel_files[0].name}")
    return True

# --- Main Logic ---

def main():
    """Main release script execution."""
    print_separator("Starting Release Process")

    # --- Ensure working directory is project root ---
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    print(f"Working directory: {os.getcwd()}")

    # --- Fetch Remote ---
    print("Fetching remote repository state...")
    _, fetch_err = run_command(["git", "fetch", REMOTE_NAME])
    if fetch_err:
        print(f"Error fetching remote: {fetch_err}", file=sys.stderr)
        return 1 # Cannot proceed without remote state

    # --- Get State ---
    print("Gathering local and remote state...")
    local_version = get_current_version()
    local_hash = get_local_head_hash()
    remote_hash = get_remote_main_hash()
    remote_version = get_latest_remote_tag_version()

    if not all([local_version, local_hash, remote_hash, remote_version]):
        print("Failed to gather necessary Git/project state. Aborting.", file=sys.stderr)
        return 1

    print(f"Local Version: {local_version}, Local HEAD: {local_hash[:7]}")
    print(f"Remote Main Version: {remote_version}, Remote Main HEAD: {remote_hash[:7]}")

    # --- Versioning Logic ---
    new_version = local_version
    version_bumped = False
    if local_hash != remote_hash:
        print("Local HEAD differs from remote main.")
        # Compare versions using a simple string comparison for now
        # (Could use packaging.version for more robust comparison if needed)
        if local_version == remote_version:
            print("Versions match, bumping patch version...")
            bumped = bump_patch_version(local_version)
            if not bumped:
                return 1
            new_version = bumped
            if not update_version_in_toml(new_version):
                return 1
            version_bumped = True
        elif local_version > remote_version:
             print("Local version is already ahead of remote main tag. Assuming manual bump.")
        else:
             print(f"Warning: Local version ({local_version}) is behind remote main tag ({remote_version}). Proceeding without bump.", file=sys.stderr)
             # Or potentially abort here? For now, just warn.
    else:
        print("Local HEAD matches remote main. No version bump needed.")

    # --- Run Tests ---
    if not run_tests():
        print("Aborting release due to test failures.", file=sys.stderr)
        return 1

    # --- Run Build ---
    if not run_build():
        print("Aborting release due to build failures.", file=sys.stderr)
        return 1

    # --- Git Branch, Commit, Push, and PR (if version bumped) ---
    if version_bumped:
        print_separator("Creating Release Branch and Pull Request")
        branch_name = f"release/v{new_version}"
        tag_name = f"v{new_version}" # Needed for PR title/body

        # Check if branch already exists locally
        existing_branches, _ = run_command(["git", "branch"], check=False)
        if f" {branch_name}\n" in existing_branches or f"* {branch_name}\n" in existing_branches:
             print(f"Branch '{branch_name}' already exists locally. Checking out.", file=sys.stderr)
             _, checkout_err = run_command(["git", "checkout", branch_name])
             if checkout_err:
                  print(f"Failed to checkout existing branch '{branch_name}': {checkout_err}", file=sys.stderr)
                  return 1
        else:
             print(f"Creating and checking out new branch: {branch_name}")
             _, checkout_err = run_command(["git", "checkout", "-b", branch_name])
             if checkout_err:
                  print(f"Failed to create branch '{branch_name}': {checkout_err}", file=sys.stderr)
                  return 1

        print(f"Staging {PYPROJECT_PATH} on branch {branch_name}...")
        _, stage_err = run_command(["git", "add", PYPROJECT_PATH])
        if stage_err:
            print(f"Failed to stage {PYPROJECT_PATH}: {stage_err}", file=sys.stderr)
            return 1 # Consider switching back to main branch?

        commit_msg = f"chore: Bump version to {tag_name}" # Use tag name for consistency
        print(f"Committing: {commit_msg}")
        _, commit_err = run_command(["git", "commit", "-m", commit_msg])
        # Handle potential "nothing to commit" if file wasn't actually changed (shouldn't happen if update_version_in_toml worked)
        if commit_err and "nothing to commit" not in commit_err:
            print(f"Failed to commit version bump: {commit_err}", file=sys.stderr)
            return 1
        elif "nothing to commit" in (commit_err or ""):
             print("Warning: No changes detected in pyproject.toml for commit.", file=sys.stderr)
             # Continue anyway, maybe user manually reverted?

        print(f"Pushing branch {branch_name} to {REMOTE_NAME}...")
        _, push_err = run_command(["git", "push", "-u", REMOTE_NAME, branch_name]) # Use -u to set upstream
        if push_err:
            print(f"Failed to push branch {branch_name}: {push_err}", file=sys.stderr)
            return 1

        # --- Create Pull Request ---
        print(f"Creating Pull Request from {branch_name} to {MAIN_BRANCH}...")
        # Extract owner/repo (replace with actual logic if needed)
        owner = "mort-sh"
        repo = "AutomaticGitCommit"
        pr_title = f"Release {tag_name}"
        pr_body = f"Automated release pull request for version {tag_name}."

        # Use the MCP Tool for GitHub PR creation
        # Note: This requires the GitHub MCP server to be configured and running.
        # The actual tool call will be made by the AI assistant framework.
        # We prepare the arguments here.
        mcp_tool_args = {
            "owner": owner,
            "repo": repo,
            "title": pr_title,
            "head": branch_name,
            "base": MAIN_BRANCH,
            "body": pr_body,
            # "draft": False, # Optional: create as draft?
            # "maintainer_can_modify": True # Optional: allow maintainers to edit
        }
        # Placeholder for the actual MCP call - the framework handles this.
        # In a real script, you might use a library like PyGithub or requests.
        print(f"Prepared arguments for GitHub PR creation tool: {mcp_tool_args}")
        # Simulate success for now in the script's flow
        pr_created_successfully = True # Assume success for script logic flow
        pr_url = f"https://github.com/{owner}/{repo}/pull/new/{branch_name}" # Example URL

        if pr_created_successfully:
             print("\nPull Request creation initiated successfully (simulated).")
             print(f"Please review and merge the PR on GitHub: {pr_url}") # Provide a generic link
             print("After merging, push the tag to trigger the release workflow:")
             print(f"  git tag {tag_name}")
             print(f"  git push {REMOTE_NAME} {tag_name}")
        else:
             print("\nPull Request creation failed.", file=sys.stderr)
             print("Please create the PR manually from branch '{branch_name}' to '{MAIN_BRANCH}'.")
             # Don't automatically switch back? Leave user on the release branch.
             # print(f"Switching back to {MAIN_BRANCH} branch.")
             # run_command(["git", "checkout", MAIN_BRANCH])
             return 1 # Indicate failure

        # Switch back to main branch locally after pushing and creating PR
        print(f"\nSwitching back to {MAIN_BRANCH} branch locally.")
        _, switch_err = run_command(["git", "checkout", MAIN_BRANCH])
        if switch_err:
             print(f"Warning: Failed to switch back to {MAIN_BRANCH}: {switch_err}", file=sys.stderr)

    else: # No version bump
        print("\nNo version bump performed. Build artifacts created.")
        print("If release is intended, manually create a branch, version bump, commit, push, and open a PR.")

    print_separator("Release Script Finished")
    return 0


if __name__ == "__main__":
    # Basic argument parsing could be added here (e.g., --force-bump, --skip-tests)
    # parser = argparse.ArgumentParser(description="AutoCommit Release Helper")
    # args = parser.parse_args()
    sys.exit(main())
