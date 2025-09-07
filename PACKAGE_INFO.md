# Stacks Agent Protocol - Package Information

## 📦 Package Structure

```
stacks_agent_protocol/
├── stacks_agent_protocol/          # Main package directory
│   └── __init__.py                 # Main SAPAgent code
├── tests/                          # Test files
│   ├── __init__.py
│   └── test_agent.py
├── dist/                           # Built packages
│   ├── stacks_agent_protocol-1.0.0-py3-none-any.whl
│   └── stacks_agent_protocol-1.0.0.tar.gz
├── setup.py                        # Legacy setup file
├── pyproject.toml                  # Modern Python packaging
├── requirements.txt                # Dependencies
├── README.md                       # Package documentation
├── LICENSE                         # MIT License
├── MANIFEST.in                     # Package manifest
├── Makefile                        # Build commands
├── example.py                      # Usage example
└── .gitignore                      # Git ignore rules
```

## 🚀 Installation Options

### 1. Install from Local Package
```bash
# Install from wheel
pip install dist/stacks_agent_protocol-1.0.0-py3-none-any.whl

# Install from source
pip install dist/stacks_agent_protocol-1.0.0.tar.gz

# Install in development mode
pip install -e .
```

### 2. Install with Dependencies
```bash
# Install with development dependencies
pip install -e ".[dev]"

# Install with Streamlit support
pip install -e ".[streamlit]"

# Install all optional dependencies
pip install -e ".[all]"
```

## 🔧 Build Commands

### Using Makefile
```bash
make help          # Show available commands
make install       # Install the package
make install-dev   # Install with dev dependencies
make test          # Run tests
make lint          # Run linting
make format        # Format code
make clean         # Clean build artifacts
make build         # Build the package
make upload        # Upload to PyPI
make example       # Run example script
```

### Using Python directly
```bash
# Build package
python -m build

# Install package
pip install -e .

# Run tests
pytest tests/

# Format code
black stacks_agent_protocol/ tests/
```

## 📋 Package Contents

### Core Files
- **`__init__.py`**: Main SAPAgent implementation with all tools and functionality
- **`setup.py`**: Legacy setup configuration
- **`pyproject.toml`**: Modern Python packaging configuration
- **`requirements.txt`**: Package dependencies

### Documentation
- **`README.md`**: Comprehensive package documentation
- **`LICENSE`**: MIT License
- **`example.py`**: Usage examples and demonstrations

### Testing
- **`tests/test_agent.py`**: Unit tests for the agent
- **`tests/__init__.py`**: Test package initialization

### Build Configuration
- **`MANIFEST.in`**: Files to include in the package
- **`Makefile`**: Build and development commands
- **`.gitignore`**: Git ignore patterns

## 🎯 Package Features

### Core Functionality
- ✅ **StacksAgentProtocol Class**: Main agent implementation
- ✅ **Tool Integration**: All 6 blockchain tools included
- ✅ **API Key Management**: Secure authentication system
- ✅ **MongoDB Integration**: Persistent storage and validation
- ✅ **Token Usage Tracking**: Monitor AI token consumption
- ✅ **Error Handling**: Comprehensive error management

### Dependencies
- **langchain**: Core AI framework
- **langchain-anthropic**: Claude AI integration
- **pymongo**: MongoDB database driver
- **python-dotenv**: Environment variable management
- **requests**: HTTP client library

### Optional Dependencies
- **streamlit**: Web dashboard support
- **plotly**: Data visualization
- **pytest**: Testing framework
- **black**: Code formatting
- **flake8**: Code linting
- **mypy**: Type checking

## 🔄 Usage Examples

### Basic Usage
```python
from stacks_agent_protocol import create_stacks_agent

# Create agent
agent = create_stacks_agent(
    sap_api_key="your-api-key",
    anthropic_api_key="your-anthropic-key"
)

# Use agent
response = agent.process_with_tools("What is my STX balance?")
print(response)
```

### Advanced Usage
```python
# Custom configuration
agent = create_stacks_agent(
    sap_api_key="your-api-key",
    anthropic_api_key="your-anthropic-key",
    model_name="claude-3-5-sonnet-20240620",
    temperature=0.7,
    validate_api_key=True
)

# Process requests
response = agent.process_transaction_request("Transfer 100 STX")
history = agent.get_transaction_history()
```

## 📊 Package Statistics

- **Package Size**: ~15KB (wheel), ~20KB (source)
- **Dependencies**: 5 core dependencies
- **Python Support**: 3.8, 3.9, 3.10, 3.11
- **License**: MIT
- **Version**: 1.0.0

## 🚀 Distribution

### PyPI Upload (Future)
```bash
# Upload to PyPI
twine upload dist/*

# Upload to Test PyPI
twine upload --repository testpypi dist/*

# Check package
twine check dist/*
```

### Installation from PyPI (Future)
```bash
pip install stacks-agent-protocol
```

## 🔧 Development

### Local Development
```bash
# Clone and setup
git clone <repository>
cd stacks_agent_protocol
pip install -e ".[dev]"

# Run tests
pytest tests/

# Format code
black stacks_agent_protocol/ tests/

# Lint code
flake8 stacks_agent_protocol/
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make changes
4. Add tests
5. Submit pull request

## 📝 Notes

- The package includes the complete SAPAgent implementation
- All blockchain tools are included and functional
- MongoDB integration is optional but recommended
- API key validation can be disabled for development
- The package is ready for PyPI distribution
- Comprehensive documentation and examples included

## 🎉 Success!

The Stacks Agent Protocol has been successfully packaged as a pip package with:
- ✅ Complete functionality
- ✅ Proper packaging structure
- ✅ Comprehensive documentation
- ✅ Test suite
- ✅ Build system
- ✅ Distribution ready

The package is now ready for installation, distribution, and use! 🚀