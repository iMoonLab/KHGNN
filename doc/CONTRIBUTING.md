# Contributing to KHGNN

Thank you for your interest in contributing to KHGNN! This guide will help you get started.

## 🔧 Development Setup

1. **Fork and Clone**
   ```bash
   git clone https://github.com/your-username/KHGNN.git
   cd KHGNN
   ```

2. **Install Development Dependencies**
   ```bash
   uv sync --group dev
   ```

3. **Install Pre-commit Hooks** (Optional but recommended)
   ```bash
   uv run pre-commit install
   ```

## 📝 Code Style

We use automated code formatting tools to maintain consistency:

### Automatic Formatting
- **Black**: Code formatting
- **isort**: Import sorting
- Files are automatically formatted on save in VS Code

### Manual Formatting
```bash
# Format all files
uv run black .

# Format specific files
uv run black filename.py
uv run isort filename.py
```

### Code Standards
- Follow PEP 8 guidelines
- Use type hints where possible
- Write docstrings for all public functions and classes
- Keep line length to 88 characters (Black's default)

## 🧪 Testing

```bash
# Run all tests
uv run pytest

# Run specific test file
uv run pytest tests/test_model.py

# Run with coverage
uv run pytest --cov=khgnn
```

## 📊 Experiments

When adding new features or modifications:

1. **Run baseline experiments** to ensure reproducibility
2. **Document performance changes** in your PR
3. **Update configs** if adding new parameters

```bash
# Run quick validation
uv run example.py

# Run full experiments
uv run trans_multi_train.py
uv run prod_multi_exp.py
```

## 🚀 Pull Request Process

1. **Create a feature branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. **Make your changes**
   - Write clear, concise commit messages
   - Include tests for new features
   - Update documentation as needed

3. **Ensure quality**
   ```bash
   # Run tests
   uv run pytest
   
   # Check for issues
   uv run black --check .
   uv run isort --check-only .
   ```

4. **Update documentation**
   - Update README if adding new features
   - Add docstrings to new functions
   - Update example scripts if needed

5. **Submit PR**
   - Provide a clear description of changes
   - Reference any related issues
   - Include before/after performance comparisons if applicable

## 🐛 Bug Reports

When reporting bugs, please include:

- **Environment details**: Python version, PyTorch version, OS
- **Reproduction steps**: Minimal code to reproduce the issue
- **Expected vs actual behavior**
- **Error messages**: Full stack traces if applicable

## 💡 Feature Requests

For new features:

- **Check existing issues** to avoid duplicates
- **Provide motivation**: Why is this feature needed?
- **Describe the interface**: How should it work?
- **Consider backwards compatibility**

## 📚 Documentation

We welcome improvements to documentation:

- **Code comments**: Explain complex algorithms
- **Docstrings**: Document all public APIs
- **README updates**: Keep examples current
- **Tutorials**: Help new users get started

## 🎯 Areas for Contribution

Current areas where contributions are especially welcome:

### Core Features
- [ ] Additional kernel types
- [ ] Memory optimization
- [ ] Multi-GPU support
- [ ] Mixed precision training

### Experiments
- [ ] New benchmark datasets
- [ ] Hyperparameter optimization
- [ ] Ablation studies
- [ ] Comparison with other methods

### Infrastructure
- [ ] Continuous integration
- [ ] Performance benchmarks
- [ ] Docker containers
- [ ] Documentation improvements

## 🤝 Community Guidelines

- **Be respectful** and constructive in discussions
- **Help others** learn and contribute
- **Follow the code of conduct**
- **Share knowledge** through issues and discussions

## 📞 Getting Help

- **Open an issue** for bugs or feature requests
- **Start a discussion** for questions or ideas
- **Check existing issues** before creating new ones

## 🏆 Recognition

Contributors will be acknowledged in:
- README contributor section
- Release notes for significant contributions
- Academic papers (for algorithmic contributions)

Thank you for helping make KHGNN better! 🎉
