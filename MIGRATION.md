# Migration to pyproject.toml

This project has been migrated from `setup.py` to `pyproject.toml` (PEP 517/518 standard).

## What Changed

- **Removed**: `setup.py`
- **Added**: `pyproject.toml` (modern Python packaging standard)
- **Benefits**:
  - Declarative configuration
  - Better tool integration (Black, Mypy, Pytest)
  - Future-proof standard

## Installation

The installation commands remain the same:

```powershell
# Editable install (recommended for development)
pip install -e .

# With dev dependencies
pip install -e ".[dev]"

# With GPU support (faiss-gpu)
pip install -e ".[gpu]"

# With diffusion models
pip install -e ".[diffusion]"
```

## For Contributors

If you had previously installed with `setup.py`, simply reinstall:

```powershell
pip uninstall research_project
pip install -e .
```

## Tool Configuration

The `pyproject.toml` now includes configuration for:

- **Black**: Code formatter (line-length=100)
- **Mypy**: Type checker
- **Pytest**: Test runner

Run these tools directly:

```powershell
# Format code
black src/ scripts/

# Type check
mypy src/

# Run tests
pytest
```

## Updating Metadata

Edit the `[project]` section in `pyproject.toml` to update:

- Author name and email
- Project description
- Repository URL
- Version number
- Dependencies

## More Information

- [PEP 517](https://peps.python.org/pep-0517/) - Build system specification
- [PEP 518](https://peps.python.org/pep-0518/) - Build system requirements
- [Setuptools pyproject.toml guide](https://setuptools.pypa.io/en/latest/userguide/pyproject_config.html)
