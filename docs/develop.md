## Install for develop

```bash
pip install -e .[dev]
```

## Test PyPI upload process

```bash
python -m build
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```
