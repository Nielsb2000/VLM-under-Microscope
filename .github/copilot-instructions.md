# Copilot Instructions for my-vscode-project

## Project Overview
- This is a minimal Python project targeting Python 3.12+, managed with a `pyproject.toml` and a local `.venv`.
- The main entry point is [main.py](main.py), which currently prints a greeting.
- Dependencies are managed via the `dependencies` field in [pyproject.toml](pyproject.toml) and locked in [uv.lock](uv.lock).

## Developer Workflows
- **Run the project:**
  - Activate the virtual environment in `.venv` if not already active.
  - Run with `python main.py`.
- **Install dependencies:**
  - Add packages to `dependencies` in [pyproject.toml](pyproject.toml), then run your preferred Python package manager (e.g., `uv`, `pip`, or `poetry`).
- **Update lockfile:**
  - Use `uv pip install -r requirements.txt` or similar to update [uv.lock](uv.lock) after changing dependencies.

## Conventions & Patterns
- All code lives in the project root; no submodules or packages yet.
- No custom build, test, or lint scripts are present.
- No configuration for CI/CD, formatting, or type checking is present.
- The project is designed to be simple and easily extensible.

## External Integrations
- Uses `requests` as the only external dependency (see [pyproject.toml](pyproject.toml)).
- No other integrations or service boundaries are present.

## Key Files
- [main.py](main.py): Main script and entry point.
- [pyproject.toml](pyproject.toml): Project metadata and dependencies.
- [uv.lock](uv.lock): Locked dependency versions.
- [README.md](README.md): Currently empty; update with project details as needed.

## Example: Adding a Feature
1. Add your function to [main.py](main.py).
2. If you need a new package, add it to `dependencies` in [pyproject.toml](pyproject.toml).
3. Update the lockfile and install dependencies.
4. Run and test your code with `python main.py`.

---
For more details on Copilot instructions, see [https://aka.ms/vscode-instructions-docs](https://aka.ms/vscode-instructions-docs).
