# Stacks Agent Protocol (SAP)

A comprehensive AI agent for interacting with the Stacks blockchain using LangChain and Claude AI. The Stacks Agent Protocol provides intelligent tools for blockchain operations, smart contract interactions, and DeFi applications.

## Features

### 🔧 Core Tools
- **Account Balance Tool**: Query STX and token balances
- **Account Assets Tool**: List all assets in an account
- **STX Inbound Tool**: Check STX inbound transfers
- **Account Transactions Tool**: Retrieve transaction history
- **Account Nonces Tool**: Get account nonce information
- **Testnet STX Faucet Tool**: Request testnet STX tokens

### 🤖 AI-Powered
- **Claude AI Integration**: Powered by Anthropic's Claude models
- **Natural Language Processing**: Interact using plain English
- **Context-Aware Responses**: Maintains conversation context
- **Tool Selection**: Automatically chooses appropriate tools

### 🔐 Security & Authentication
- **API Key Management**: Secure API key validation
- **MongoDB Integration**: Persistent storage and validation
- **Token Usage Tracking**: Monitor and log AI token consumption
- **User Management**: Multi-user support with individual API keys

### 📊 Analytics & Monitoring
- **Usage Metrics**: Track API calls and token usage
- **Real-time Dashboard**: Web-based monitoring interface
- **Token Analytics**: Detailed usage statistics and visualizations
- **Performance Monitoring**: Track response times and success rates

## Installation

### From PyPI (Recommended)
```bash
pip install stacks-agent-protocol
```

### From Source
```bash
git clone https://github.com/stacks-agent-protocol/stacks-agent-protocol.git
cd stacks-agent-protocol
pip install -e .
```

## Quick Start

### 1. Set up Environment Variables
```bash
export ANTHROPIC_API_KEY="your-anthropic-api-key"
export SAP_API_KEY="your-sap-api-key"
export MONGODB_URI="your-mongodb-connection-string"
```

### 2. Basic Usage
```python
from stacks_agent_protocol import create_stacks_agent

# Create an agent instance
agent = create_stacks_agent(
    sap_api_key="your-api-key",
    anthropic_api_key="your-anthropic-key",
    validate_api_key=True
)

# Process a request
response = agent.process_with_tools("What is the balance of SP3K8BC0PPEVCV7NZ6QSRWPQ2JE9E5B6N3PA0KBR9?")
print(response)
```

### 3. Advanced Usage
```python
# Create agent with custom configuration
agent = create_stacks_agent(
    sap_api_key="your-api-key",
    anthropic_api_key="your-anthropic-key",
    model_name="claude-3-5-sonnet-20240620",
    temperature=0.7,
    validate_api_key=True
)

# Process transaction requests
response = agent.process_transaction_request("Transfer 100 STX to SP3K8BC0PPEVCV7NZ6QSRWPQ2JE9E5B6N3PA0KBR9")
print(response)

# Get transaction history
history = agent.get_transaction_history()
print(history)
```

## Configuration

### Environment Variables
- `ANTHROPIC_API_KEY`: Your Anthropic API key for Claude AI
- `SAP_API_KEY`: Your Stacks Agent Protocol API key
- `MONGODB_URI`: MongoDB connection string for data storage

### Agent Parameters
- `sap_api_key`: Stacks Agent Protocol API key
- `anthropic_api_key`: Anthropic API key (optional if set in environment)
- `model_name`: Claude model to use (default: "claude-3-5-sonnet-20240620")
- `temperature`: AI response creativity (0.0-1.0, default: 0.7)
- `validate_api_key`: Whether to validate API key against database (default: True)

## API Reference

### Core Classes

#### `StacksAgentProtocol`
Main agent class for interacting with the Stacks blockchain.

**Methods:**
- `process_with_tools(request)`: Process requests using available tools
- `process_transaction_request(request)`: Handle transaction-specific requests
- `get_transaction_history()`: Retrieve conversation history
- `close_connections()`: Clean up database connections

#### `create_stacks_agent()`
Convenience function to create a configured agent instance.

**Parameters:**
- `sap_api_key`: API key for authentication
- `anthropic_api_key`: Anthropic API key
- `model_name`: Claude model name
- `temperature`: Response creativity level
- `validate_api_key`: Enable API key validation

### Tools

#### Account Balance Tool
```python
# Query STX balance
response = agent.process_with_tools("What is my STX balance?")
```

#### Account Assets Tool
```python
# List all assets
response = agent.process_with_tools("Show me all my assets")
```

#### Transaction History Tool
```python
# Get transaction history
response = agent.process_with_tools("Show my recent transactions")
```

## Web Dashboard

The package includes a web-based dashboard for monitoring and management:

### Features
- **API Key Management**: Create, view, and manage API keys
- **Usage Analytics**: Real-time token usage statistics
- **Performance Metrics**: Response times and success rates
- **User Management**: Multi-user support and access control

### Running the Dashboard
```bash
# Install with dashboard dependencies
pip install stacks-agent-protocol[streamlit]

# Run the dashboard
streamlit run dashboard.py
```

## Examples

### Basic Blockchain Queries
```python
# Check account balance
response = agent.process_with_tools("What is the balance of SP3K8BC0PPEVCV7NZ6QSRWPQ2JE9E5B6N3PA0KBR9?")

# Get transaction history
response = agent.process_with_tools("Show me the last 10 transactions for SP3K8BC0PPEVCV7NZ6QSRWPQ2JE9E5B6N3PA0KBR9")

# Check STX inbound transfers
response = agent.process_with_tools("Are there any pending STX transfers to my account?")
```

### Smart Contract Interactions
```python
# Interact with smart contracts
response = agent.process_with_tools("Call the transfer function on contract SP3K8BC0PPEVCV7NZ6QSRWPQ2JE9E5B6N3PA0KBR9 with parameters...")

# Query contract state
response = agent.process_with_tools("What is the current state of contract SP3K8BC0PPEVCV7NZ6QSRWPQ2JE9E5B6N3PA0KBR9?")
```

### DeFi Operations
```python
# Check DeFi positions
response = agent.process_with_tools("Show me my DeFi positions and yields")

# Analyze liquidity pools
response = agent.process_with_tools("What are the current APYs for STX liquidity pools?")
```

## Development

### Setting up Development Environment
```bash
# Clone the repository
git clone https://github.com/stacks-agent-protocol/stacks-agent-protocol.git
cd stacks-agent-protocol

# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black stacks_agent_protocol/

# Lint code
flake8 stacks_agent_protocol/
```

### Contributing
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Support

- **Documentation**: [https://docs.stacksagentprotocol.com](https://docs.stacksagentprotocol.com)
- **Issues**: [GitHub Issues](https://github.com/stacks-agent-protocol/stacks-agent-protocol/issues)
- **Discord**: [Stacks Agent Protocol Community](https://discord.gg/stacks-agent-protocol)
- **Email**: support@stacksagentprotocol.com

## Changelog

### Version 1.0.0
- Initial release
- Core agent functionality
- Tool integration
- API key management
- Usage tracking and analytics
- Web dashboard

## Roadmap

- [ ] Additional blockchain tools
- [ ] Smart contract deployment support
- [ ] Advanced analytics and reporting
- [ ] Mobile SDK
- [ ] Enterprise features
- [ ] Multi-chain support

---

**Built with ❤️ for the Stacks ecosystem**
