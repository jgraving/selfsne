# Contributing

Thank you for your interest in `selfsne`! This document covers how to get
started, submit changes, and follow project conventions.

## Development setup

```bash
# Clone the repository
git clone https://github.com/jgraving/selfsne
cd selfsne

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install in editable mode with dev dependencies
pip install -e '.[dev]'
```

## Running tests

```bash
pytest
```

Tests are in the `tests/` directory. Add tests for new functionality. Tests
can be marked with `@pytest.mark.cuda` to run only on hardware with CUDA.

## Code style

This project uses [Ruff](https://github.com/astral-sh/ruff) for linting:

```bash
ruff check
```

The active lint rules are: `E9` (syntax error), `F63` (comparison to
literal), `F7` (redundant return), `F82` (undefined name).

## Making changes

1. Create a feature branch from `main`:
   ```bash
   git checkout -b feature/my-feature
   ```
2. Make your changes. Add tests for new functionality.
3. Run tests and linting:
   ```bash
   pytest
   ruff check
   ```
4. Commit your changes with a clear, descriptive message.
5. Push your branch and open a pull request.

## Pull request guidelines

- Keep PRs focused on a single concern.
- Include tests for new functionality.
- Update documentation if you change public APIs.
- Ensure all CI checks pass.

## Reporting issues

Please open an issue on GitHub with:

- A clear title and description.
- Steps to reproduce the problem.
- Your environment (Python version, PyTorch version, OS).
- A minimal code example that demonstrates the issue, if possible.
