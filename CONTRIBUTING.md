# Contributing to Fine-Tuning Small LLMs

Thank you for your interest in contributing to this project! This guide will help you get started.

## 🤝 How to Contribute

### Reporting Issues
- Use the GitHub issue tracker to report bugs
- Provide clear reproduction steps
- Include environment details (OS, Python version, GPU specs)

### Suggesting Enhancements
- Open an issue with the "enhancement" label
- Describe the feature and its benefits
- Provide implementation ideas if possible

### Code Contributions
1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Commit with clear messages
7. Push to your fork
8. Open a Pull Request

## 🛠️ Development Setup

```bash
# Clone your fork
git clone https://github.com/your-username/fine-tuning-small-llms.git
cd fine-tuning-small-llms

# Install development dependencies
pip install -r requirements-dev.txt

# Run tests
pytest tests/

# Format code
black src/
isort src/

# Type checking
mypy src/
```

## 📋 Code Standards

### Python Code Style
- Follow PEP 8 guidelines
- Use Black for code formatting
- Use isort for import organization
- Add type hints where applicable
- Write docstrings for functions and classes

### Documentation
- Update README files for significant changes
- Add inline comments for complex logic
- Update blog post references if needed

### Testing
- Write unit tests for new functionality
- Ensure all tests pass before submitting
- Aim for good test coverage

## 🔄 Pull Request Process

1. **Before Opening a PR**:
   - Ensure your code follows the style guidelines
   - Run all tests and ensure they pass
   - Update documentation as needed
   - Rebase your branch on the latest main

2. **PR Description**:
   - Clearly describe what changes you made
   - Reference any related issues
   - Include screenshots for UI changes
   - List any breaking changes

3. **Review Process**:
   - All PRs require review before merging
   - Address review feedback promptly
   - Keep discussions focused and constructive

## 📝 Commit Message Guidelines

Use clear, descriptive commit messages:

```
feat: add support for Llama 3.2 models
fix: resolve memory leak in training loop
docs: update installation instructions
refactor: simplify dataset creation logic
```

Prefixes:
- `feat:` - New features
- `fix:` - Bug fixes
- `docs:` - Documentation changes
- `refactor:` - Code refactoring
- `test:` - Adding or updating tests
- `chore:` - Maintenance tasks

## 🏷️ Issue Labels

- `bug` - Something isn't working
- `enhancement` - New feature or request
- `documentation` - Improvements to docs
- `good first issue` - Good for newcomers
- `help wanted` - Extra attention needed
- `question` - Further information requested

## 📚 Areas for Contribution

### High Priority
- Additional model support (Gemma, Qwen, etc.)
- Performance optimizations
- Documentation improvements
- Example datasets for different domains

### Medium Priority
- UI/UX improvements for web interfaces
- Additional deployment options (Kubernetes, cloud platforms)
- Cost optimization tools
- Integration with more monitoring tools

### Nice to Have
- Mobile app interfaces
- Voice interface integration
- Advanced visualization tools
- Multi-language support

## 🧪 Testing Guidelines

### Unit Tests
- Test individual functions and classes
- Mock external dependencies
- Focus on edge cases and error conditions

### Integration Tests
- Test component interactions
- Use test databases and services
- Verify end-to-end workflows

### Performance Tests
- Benchmark critical paths
- Monitor memory usage
- Test with different model sizes

## 📖 Documentation

### Code Documentation
- Use clear, descriptive docstrings
- Include parameter descriptions and return values
- Provide usage examples

### User Documentation
- Keep README files up to date
- Include setup and usage instructions
- Provide troubleshooting guides

## 🚀 Release Process

1. Update version numbers
2. Update CHANGELOG.md
3. Create release notes
4. Tag the release
5. Build and test release artifacts
6. Publish to package repositories

## 💬 Community

- Be respectful and constructive
- Help newcomers get started
- Share knowledge and best practices
- Follow the code of conduct

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

## 🙏 Recognition

Contributors will be recognized in:
- README.md acknowledgments
- Release notes
- Blog post updates (where appropriate)

Thank you for helping make this project better! 🌟
