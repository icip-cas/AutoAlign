# AutoAlign Development Guide

## Fork and Clone the Repository

First, fork this repository and then clone your forked repository using the following commands:

```bash
git clone [your-repository-url]
cd AutoAlign
```

## Development Environment Setup

To set up your development environment, run:

```bash
pip install -e .[dev]
```

## Development Process

Create a new branch for your feature development using:

```bash
git checkout -b feature/your-feature-name
```

To maintain consistency and organization in the project, make sure to follow the branch naming conventions. The available branch types are listed below:

### Branch Naming Conventions

Adhere to the following Git branch naming conventions to maintain clarity and organization:

| Branch Type        | Naming Pattern  | Description                                      |
|--------------------|-----------------|--------------------------------------------------|
| Main Branch        | main/master     | Primary branch for stable releases               |
| Development Branch | develop/dev     | Branch for ongoing development                   |
| Feature Branch     | feature/*       | Branches for developing new features             |
| Fix Branch         | fix/*           | Branches for bug fixes                           |
| Hotfix Branch      | hotfix/*        | Branches for urgent production fixes             |
| Release Branch     | release/*       | Branches for preparing version releases          |
| Test Branch        | test/*          | Branches for testing and validation              |
| Experiment Branch  | experiment/*    | Branches for trying out experimental features    |
| Documentation Branch| docs/*         | Branches for updating documentation              |
| Refactor Branch    | refactor/*      | Branches for code refactoring                    |

## Adding Tests

Place your test files in the `tests` directory, following the naming pattern `test_*.py`. Write test cases using pytest that cover new features or bug fixes, including both standard and edge cases. Refer to existing tests for guidance.

## Running Tests

To execute your tests, use:

```bash
python -m pytest
```

## Submitting a Pull Request

Follow the guidelines in `.github/PULL_REQUEST_TEMPLATE.md` to submit a pull request.

## Upload Process to Test PyPI (For Maintainers)

To test the PyPI upload process, execute:

```bash
python -m build
twine upload --repository-url https://test.pypi.org/legacy/ dist/*
```

By following this guide, you ensure a streamlined and organized development process for AutoAlign.
