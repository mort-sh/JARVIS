# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build/Test/Lint Commands
- Build: `python -m scripts.build` or `uv run project_build`
- Clean: `python -m scripts.clean` or `uv run project_clean`
- Lint: `ruff check .` (verify syntax and conventions)
- Format: `ruff format .` (automatically fix formatting issues)
- Run tests: `uv run pytest -v` or `python -m pytest`
- Run single test: `uv run pytest tests/test_specific.py::test_function`

## Coding Guidelines
- Python version: >= 3.12
- Line length: 100 characters maximum
- Quote style: Double quotes
- Indentation: 4 spaces (not tabs)
- Imports: Organized in groups (stdlib → third-party → local), sorted alphabetically
- Naming: Classes=PascalCase, functions/variables=snake_case, constants=UPPER_SNAKE_CASE
- Private members: Prefix with underscore (_private_method)
- Type hints: Required for functions/methods (using typing module)
- Error handling: Catch specific exceptions, use detailed error messages, provide fallbacks
- Docstrings: Required for modules, classes, methods with Args/Returns/Raises sections
- Always run linter and formatter before committing changes

When contributing code, follow the formatting and style guide in pyproject.toml.