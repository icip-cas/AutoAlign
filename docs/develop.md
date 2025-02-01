# Develop Guidance

## Fork and clone this repo

```bash
git clone [your-repository-url]
cd AutoAlign
```

## Install for develop

```bash
pip install -e .[dev]
```

## Develop

Create feature branches and develop a feature:

```bash
git checkout -b feature/your-feature-name
```

## Run the tests

```bash
python -m pytest
```

## Test PyPI upload process

```bash
python -m build
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```
