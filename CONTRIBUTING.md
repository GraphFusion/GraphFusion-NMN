
# Contributing to GraphFusion

We welcome contributions to the **GraphFusion** project! Whether you're interested in improving the core features, adding new examples, fixing bugs, or improving documentation, your contributions are highly appreciated.

## Table of Contents

- [Code of Conduct](#code-of-conduct)
- [How to Contribute](#how-to-contribute)
  - [Reporting Issues](#reporting-issues)
  - [Feature Requests](#feature-requests)
  - [Submitting a Pull Request](#submitting-a-pull-request)
- [Development Setup](#development-setup)
  - [Forking the Repository](#forking-the-repository)
  - [Setting Up Your Development Environment](#setting-up-your-development-environment)
  - [Running Tests](#running-tests)
- [Code Style and Best Practices](#code-style-and-best-practices)
- [License](#license)

## Code of Conduct

By participating in this project, you agree to abide by our [Code of Conduct](CODE_OF_CONDUCT.md).

## How to Contribute

### Reporting Issues

If you encounter a bug or problem with **GraphFusion**, please open an issue in the [Issues section](https://github.com/GraphFusion/GraphFusion-NMN/issues) of the repository. Make sure to provide:

- A clear description of the issue
- Steps to reproduce the issue
- Expected and actual behavior
- Any relevant error messages or logs

### Feature Requests

If you have an idea for a new feature or an improvement to an existing one, feel free to open an issue and label it as a **Feature Request**. Be as specific as possible so that the maintainers can evaluate and prioritize your suggestion.

### Submitting a Pull Request

If you want to contribute code, please follow these steps:

1. Fork the repository.
2. Create a new branch from `main` for your feature or bug fix. Use a descriptive name for the branch, e.g., `feature/new-functionality` or `bugfix/fix-crash`.
3. Write clear, concise commit messages following [best commit practices](https://chris.beams.io/posts/git-commit/).
4. Ensure that your code adheres to the project's coding standards and passes all tests.
5. Submit a pull request (PR) to the `main` branch. Please include a detailed description of your changes and reference the relevant issue (if applicable).

We will review your pull request and get back to you as soon as possible.

## Development Setup

### Forking the Repository

1. Navigate to the [GraphFusion repository](https://github.com/GraphFusion/GraphFusion-NMN).
2. Click the **Fork** button at the top right of the page.
3. Clone your forked repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/GraphFusion-NMN.git
   cd GraphFusion-NMN
   ```

### Setting Up Your Development Environment

1. **Install dependencies**:

   Make sure you have Python 3.8 or higher installed. You can use [virtual environments](https://docs.python.org/3/library/venv.html) to manage dependencies.

   ```bash
   pip install -e .
   ```

2. **Install additional development dependencies** (e.g., for testing or linting):

   ```bash
   pip install -r requirements-dev.txt
   ```

### Running Tests

To ensure your changes don’t break existing functionality, you should run the tests before submitting a pull request.

1. **Run tests**:

   ```bash
   pytest
   ```

   Ensure that all tests pass before submitting your PR.

## Code Style and Best Practices

Please follow these guidelines to maintain the quality of the project:

- **Code Formatting**: We use `black` for code formatting. Please make sure your code is formatted before submitting it.
  
  To format your code with `black`, run:

  ```bash
  black .
  ```

- **Docstrings**: Ensure that your functions, methods, and classes have proper docstrings explaining their purpose, parameters, and return values. Follow the [PEP 257 conventions](https://www.python.org/dev/peps/pep-0257/).
  
- **Tests**: Always write tests to cover your changes. Tests should be placed in the `tests/` directory. If you're fixing a bug, ensure that you write a test to verify the issue has been fixed.

## License

By contributing to **GraphFusion**, you agree that your contributions will be licensed under the [Apache 2.0 License](LICENSE).

