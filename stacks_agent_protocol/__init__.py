"""
Stacks Agent Protocol (SAP) using LangChain with Claude API

A powerful agent framework for building onchain transactions on Stacks blockchain,
leveraging AI capabilities for intelligent transaction processing and blockchain interactions.
"""

import os
import requests
import hashlib
from typing import Optional, Dict, Any, List
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain.schema import HumanMessage, SystemMessage, AIMessage
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.tools import BaseTool
from langchain.agents import AgentExecutor
from pydantic import BaseModel, Field
from pymongo import MongoClient
from datetime import datetime

# Load environment variables
load_dotenv()


# Tool Input Models
class AccountBalanceInput(BaseModel):
    """Input for getting account balance."""
    principal: str = Field(..., description="The Stacks address to check balance for")
    network: str = Field(default="testnet", description="Network to query: 'mainnet' or 'testnet'")


class AccountAssetsInput(BaseModel):
    """Input for getting account assets."""
    principal: str = Field(..., description="The Stacks address to query")
    network: str = Field(default="testnet", description="Network to query: 'mainnet' or 'testnet'")
    limit: int = Field(default=50, description="Maximum number of assets to return")
    offset: int = Field(default=0, description="Number of assets to skip")
    unanchored: bool = Field(default=True, description="Include unanchored transactions")
    until_block: Optional[str] = Field(default=None, description="Query up to specific block")


class STXInboundInput(BaseModel):
    """Input for getting STX inbound transfers."""
    principal: str = Field(..., description="The Stacks address to query")
    network: str = Field(default="testnet", description="Network to query: 'mainnet' or 'testnet'")
    limit: int = Field(default=50, description="Maximum number of transactions to return")
    offset: int = Field(default=0, description="Number of transactions to skip")
    height: Optional[int] = Field(default=None, description="Filter by block height")
    unanchored: bool = Field(default=True, description="Include unanchored transactions")
    until_block: Optional[str] = Field(default=None, description="Query up to specific block")


class AccountTransactionsInput(BaseModel):
    """Input for getting account transactions."""
    principal: str = Field(..., description="The Stacks address to query")
    network: str = Field(default="testnet", description="Network to query: 'mainnet' or 'testnet'")
    limit: int = Field(default=50, description="Maximum number of transactions to return")
    offset: int = Field(default=0, description="Number of transactions to skip")
    height: Optional[int] = Field(default=None, description="Filter by block height")
    unanchored: bool = Field(default=True, description="Include unanchored transactions")
    until_block: Optional[str] = Field(default=None, description="Query up to specific block")
    exclude_function_args: bool = Field(default=False, description="Exclude function arguments from results")


class AccountNoncesInput(BaseModel):
    """Input for getting account nonces."""
    principal: str = Field(..., description="The Stacks address to query")
    network: str = Field(default="testnet", description="Network to query: 'mainnet' or 'testnet'")
    block_height: Optional[int] = Field(default=None, description="Block height to query")
    block_hash: Optional[str] = Field(default=None, description="Block hash to query")


class TestnetSTXFaucetInput(BaseModel):
    """Input for requesting testnet STX from faucet."""
    address: str = Field(..., description="The Stacks address to send testnet STX to")
    stacking: bool = Field(default=False, description="Whether the address will be used for stacking")


# Tool Classes
class AccountBalanceTool(BaseTool):
    """Tool for getting Stacks account balance information."""
    name: str = "get_account_balance"
    description: str = "Get the balance of a Stacks account including STX and token balances"
    args_schema: type = AccountBalanceInput

    def _run(self, principal: str, network: str = "testnet") -> Dict[str, Any]:
        """Get account balance using Hiro API."""
        # Strip whitespace from principal
        principal = principal.strip()
        
        # Determine the base URL based on network
        if network == "mainnet":
            base_url = "https://api.hiro.so/extended"
        else:  # testnet
            base_url = "https://api.testnet.hiro.so/extended"
        
        # Construct the API endpoint
        url = f"{base_url}/v1/address/{principal}/balances"
        headers = {
            'Accept': 'application/json'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
            
            balance_data = response.json()
            
            # Process and format the balance data
            formatted_balance = {
                "address": principal,
                "network": network,
                "stx_balance": {
                    "balance": int(balance_data.get("stx", {}).get("balance", 0)),
                    "balance_stx": float(balance_data.get("stx", {}).get("balance", 0)) / 1000000,
                    "locked": int(balance_data.get("stx", {}).get("locked", 0)),
                    "locked_stx": float(balance_data.get("stx", {}).get("locked", 0)) / 1000000,
                    "unlock_height": balance_data.get("stx", {}).get("unlock_height", 0),
                    "burnchain_lock_height": balance_data.get("stx", {}).get("burnchain_lock_height", 0),
                    "burnchain_unlock_height": balance_data.get("stx", {}).get("burnchain_unlock_height", 0)
                },
                "fungible_tokens": balance_data.get("fungible_tokens", {}),
                "non_fungible_tokens": balance_data.get("non_fungible_tokens", {})
            }
            
            return formatted_balance
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Error fetching balance for {principal}: {str(e)}"
            return {
                "error": error_msg,
                "address": principal,
                "network": network
            }


class AccountAssetsTool(BaseTool):
    """Tool for getting all assets held by a Stacks account."""
    name: str = "get_account_assets"
    description: str = "Get all assets (fungible and non-fungible tokens) held by a Stacks account"
    args_schema: type = AccountAssetsInput

    def _run(self, principal: str, network: str = "testnet", limit: int = 50, offset: int = 0,
             unanchored: bool = True, until_block: str = None) -> Dict[str, Any]:
        """Get account assets using Hiro API."""
        # Strip whitespace from principal
        principal = principal.strip()
        
        # Determine the base URL based on network
        if network == "mainnet":
            base_url = "https://api.hiro.so/extended"
        else:  # testnet
            base_url = "https://api.testnet.hiro.so/extended"
        
        # Construct the API endpoint
        url = f"{base_url}/v1/address/{principal}/assets"
        
        # Build query parameters
        params = {
            "limit": limit,
            "offset": offset,
            "unanchored": str(unanchored).lower()
        }
        if until_block:
            params["until_block"] = until_block
        
        headers = {'Accept': 'application/json'}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            assets_data = response.json()
            
            return {
                "address": principal,
                "network": network,
                "total_assets": assets_data.get("total", 0),
                "limit": limit,
                "offset": offset,
                "results": assets_data.get("results", [])
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Error fetching assets for {principal}: {str(e)}",
                "address": principal,
                "network": network
            }


class STXInboundTool(BaseTool):
    """Tool for getting inbound STX transfers for a Stacks account."""
    name: str = "get_stx_inbound"
    description: str = "Get inbound STX transfers for a Stacks account"
    args_schema: type = STXInboundInput

    def _run(self, principal: str, network: str = "testnet", limit: int = 50, offset: int = 0,
             height: int = None, unanchored: bool = True, until_block: str = None) -> Dict[str, Any]:
        """Get inbound STX transfers using Hiro API."""
        # Strip whitespace from principal
        principal = principal.strip()
        
        # Determine the base URL based on network
        if network == "mainnet":
            base_url = "https://api.hiro.so/extended"
        else:  # testnet
            base_url = "https://api.testnet.hiro.so/extended"
        
        # Construct the API endpoint
        url = f"{base_url}/v1/address/{principal}/stx_inbound"
        
        # Build query parameters
        params = {
            "limit": limit,
            "offset": offset,
            "unanchored": str(unanchored).lower()
        }
        if height:
            params["height"] = height
        if until_block:
            params["until_block"] = until_block
        
        headers = {'Accept': 'application/json'}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            inbound_data = response.json()
            
            return {
                "address": principal,
                "network": network,
                "total_inbound": inbound_data.get("total", 0),
                "limit": limit,
                "offset": offset,
                "results": inbound_data.get("results", [])
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Error fetching inbound STX for {principal}: {str(e)}",
                "address": principal,
                "network": network
            }


class AccountTransactionsTool(BaseTool):
    """Tool for getting transaction history for a Stacks account."""
    name: str = "get_account_transactions"
    description: str = "Get transaction history for a Stacks account"
    args_schema: type = AccountTransactionsInput

    def _run(self, principal: str, network: str = "testnet", limit: int = 50, offset: int = 0,
             height: int = None, unanchored: bool = True, until_block: str = None,
             exclude_function_args: bool = False) -> Dict[str, Any]:
        """Get account transactions using Hiro API."""
        # Strip whitespace from principal
        principal = principal.strip()
        
        # Determine the base URL based on network
        if network == "mainnet":
            base_url = "https://api.hiro.so/extended"
        else:  # testnet
            base_url = "https://api.testnet.hiro.so/extended"
        
        # Construct the API endpoint
        url = f"{base_url}/v1/address/{principal}/transactions"
        
        # Build query parameters
        params = {
            "limit": limit,
            "offset": offset,
            "unanchored": str(unanchored).lower(),
            "exclude_function_args": str(exclude_function_args).lower()
        }
        if height:
            params["height"] = height
        if until_block:
            params["until_block"] = until_block
        
        headers = {'Accept': 'application/json'}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            tx_data = response.json()
            
            return {
                "address": principal,
                "network": network,
                "total_transactions": tx_data.get("total", 0),
                "limit": limit,
                "offset": offset,
                "results": tx_data.get("results", [])
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Error fetching transactions for {principal}: {str(e)}",
                "address": principal,
                "network": network
            }


class AccountNoncesTool(BaseTool):
    """Tool for getting nonce information for a Stacks account."""
    name: str = "get_account_nonces"
    description: str = "Get nonce information for a Stacks account at a specific block"
    args_schema: type = AccountNoncesInput

    def _run(self, principal: str, network: str = "testnet", block_height: int = None,
             block_hash: str = None) -> Dict[str, Any]:
        """Get account nonces using Hiro API."""
        # Strip whitespace from principal
        principal = principal.strip()
        
        # Determine the base URL based on network
        if network == "mainnet":
            base_url = "https://api.hiro.so/extended"
        else:  # testnet
            base_url = "https://api.testnet.hiro.so/extended"
        
        # Construct the API endpoint
        url = f"{base_url}/v1/address/{principal}/nonces"
        
        # Build query parameters
        params = {}
        if block_height:
            params["block_height"] = block_height
        if block_hash:
            params["block_hash"] = block_hash
        
        headers = {'Accept': 'application/json'}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            nonce_data = response.json()
            
            return {
                "address": principal,
                "network": network,
                "nonce_data": nonce_data
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Error fetching nonces for {principal}: {str(e)}",
                "address": principal,
                "network": network
            }


class TestnetSTXFaucetTool(BaseTool):
    """Tool for requesting testnet STX from the faucet."""
    name: str = "request_testnet_stx"
    description: str = "Request testnet STX from the faucet (testnet only)"
    args_schema: type = TestnetSTXFaucetInput

    def _run(self, address: str, stacking: bool = False) -> Dict[str, Any]:
        """Request testnet STX from faucet using Hiro API."""
        # Strip whitespace from address
        address = address.strip()
        
        # This endpoint only works on testnet
        base_url = "https://api.testnet.hiro.so/extended"
        url = f"{base_url}/v1/faucets/stx"
        
        # Build query parameters
        params = {
            "address": address,
            "stacking": str(stacking).lower()
        }
        
        headers = {'Accept': 'application/json'}
        
        try:
            response = requests.post(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            
            faucet_data = response.json()
            
            return {
                "address": address,
                "stacking": stacking,
                "success": True,
                "response": faucet_data
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Error requesting testnet STX for {address}: {str(e)}",
                "address": address,
                "stacking": stacking,
                "success": False
            }


class StacksAgentProtocol:
    """
    Stacks Agent Protocol - An AI-powered agent framework for building onchain transactions on Stacks.
    
    This agent combines Claude's intelligence with Stacks blockchain capabilities to:
    - Analyze and construct Stacks transactions
    - Provide smart contract interaction guidance
    - Generate Clarity code suggestions
    - Assist with DeFi operations on Stacks
    """
    
    def __init__(
        self, 
        model_name: str = "claude-3-5-sonnet-20240620",
        temperature: float = 0.7,
        system_message: Optional[str] = None,
        api_key: Optional[str] = None,
        sap_api_key: Optional[str] = None,
        validate_api_key: bool = True
    ):
        """
        Initialize the Stacks Agent Protocol.
        
        Args:
            model_name: The Claude model to use (default: claude-3-5-sonnet-20240620)
            temperature: Temperature for response generation (0.0-1.0)
            system_message: Optional system message to set agent behavior
            api_key: Anthropic API key (will use ANTHROPIC_API_KEY env var if not provided)
            sap_api_key: SAP Agent Protocol API key for authentication
            validate_api_key: Whether to validate the SAP API key (default: True)
        """
        self.model_name = model_name
        self.temperature = temperature
        self.system_message = system_message or self._get_default_system_message()
        self.sap_api_key = sap_api_key or os.getenv("SAP_API_KEY")
        
        # Initialize MongoDB connection for API key validation and token logging
        self.mongodb_client = None
        self.api_keys_collection = None
        self._init_mongodb_connection()  # Always initialize for token logging
        
        # Validate SAP API key if required
        if validate_api_key:
            if not self.sap_api_key:
                raise ValueError("SAP Agent Protocol API key is required. Please provide sap_api_key parameter or set SAP_API_KEY environment variable.")
            
            if not self._validate_sap_api_key():
                raise ValueError("Invalid SAP Agent Protocol API key. Please check your API key or generate a new one from the dashboard.")
        
        # Set up Anthropic API key
        if api_key:
            os.environ["ANTHROPIC_API_KEY"] = api_key
        elif not os.getenv("ANTHROPIC_API_KEY"):
            raise ValueError("Anthropic API key must be provided either as parameter or ANTHROPIC_API_KEY environment variable")
        
        # Initialize the language model
        self.llm = ChatAnthropic(
            model=self.model_name,
            temperature=self.temperature
        )
        
        # Initialize tools
        self.tools = [
            AccountBalanceTool(),
            AccountAssetsTool(),
            STXInboundTool(),
            AccountTransactionsTool(),
            AccountNoncesTool(),
            TestnetSTXFaucetTool()
        ]
        
        # Initialize transaction history and blockchain context
        self.transaction_history = []
        self.blockchain_context = {
            "network": "testnet",  # or mainnet
            "current_block": None,
            "stx_price": None,
            "active_contracts": []
        }
    
    def _init_mongodb_connection(self):
        """Initialize MongoDB connection for API key validation"""
        try:
            # Get MongoDB connection string from environment or use default
            mongodb_uri = os.getenv("MONGODB_URI", "mongodb+srv://testuser:HelloWorld12345@cluster0.qi14uvo.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0")
            
            self.mongodb_client = MongoClient(mongodb_uri)
            db = self.mongodb_client.stacks_agent_landing
            self.api_keys_collection = db.api_keys
            
            # Test connection
            self.mongodb_client.admin.command('ping')
            
        except Exception as e:
            print(f"Warning: Could not connect to MongoDB for API key validation: {e}")
            self.mongodb_client = None
            self.api_keys_collection = None
    
    def _hash_api_key(self, api_key: str) -> str:
        """Hash an API key for comparison with stored hashes"""
        return hashlib.sha256(api_key.encode()).hexdigest()
    
    def _log_token_usage(self, user_tokens: int, response_tokens: int, api_key: str = None):
        """Log token usage to MongoDB for metrics"""
        try:
            if self.mongodb_client is None:
                return
            
            # Get the database and collection
            db = self.mongodb_client.stacks_agent_landing
            usage_collection = db.token_usage
            
            # Find the API key document to get user_id
            user_id = None
            hashed_key = None
            if api_key:
                hashed_key = self._hash_api_key(api_key)
                key_doc = self.api_keys_collection.find_one({'hashed_key': hashed_key})
                if key_doc:
                    user_id = key_doc.get('user_id')
            
            # Log the usage
            usage_doc = {
                'user_id': user_id,
                'api_key_hash': hashed_key,
                'user_tokens': user_tokens,
                'response_tokens': response_tokens,
                'total_tokens': user_tokens + response_tokens,
                'timestamp': datetime.utcnow(),
                'date': datetime.utcnow().strftime('%Y-%m-%d')
            }
            
            usage_collection.insert_one(usage_doc)
            
        except Exception as e:
            print(f"Warning: Could not log token usage: {e}")
    
    def _validate_sap_api_key(self) -> bool:
        """
        Validate the SAP API key against MongoDB database.
        
        Returns:
            bool: True if API key is valid, False otherwise
        """
        try:
            if self.api_keys_collection is None:
                print("Warning: MongoDB connection not available for API key validation")
                return True  # Allow operation if MongoDB is not available
            
            # Hash the provided API key
            hashed_key = self._hash_api_key(self.sap_api_key)
            print(f"Hashed key: {hashed_key}")
            # Look up the hashed key in the database
            key_doc = self.api_keys_collection.find_one({
                'hashed_key': hashed_key,
                'active': True
            })
            print(f"Key doc: {key_doc}")
            if key_doc:
                # Update last used timestamp
                self.api_keys_collection.update_one(
                    {'_id': key_doc['_id']},
                    {'$set': {'last_used': datetime.utcnow()}}
                )
                return True
            else:
                return False
                
        except Exception as e:
            print(f"Warning: API key validation error: {e}")
            return False
        
    def _get_default_system_message(self) -> str:
        """Get the default system message for Stacks Agent Protocol."""
        return """You are a Stacks Agent Protocol (SAP) - an expert AI agent specialized in Stacks blockchain operations.

Your capabilities include:
- Analyzing and constructing Stacks transactions
- Providing guidance on smart contract interactions
- Generating Clarity smart contract code
- Assisting with DeFi operations (STX staking, Bitcoin DeFi, etc.)
- Explaining Stacks blockchain concepts and mechanics
- Helping with wallet operations and transaction optimization
- Real-time blockchain data retrieval using specialized tools

You have access to the following tools for blockchain interactions:
- get_account_balance: Retrieve STX and token balances for any Stacks address
- get_account_assets: Get all fungible and non-fungible tokens held by an address
- get_stx_inbound: View inbound STX transfers for an address
- get_account_transactions: Access transaction history for any address
- get_account_nonces: Get nonce information for transaction building
- request_testnet_stx: Request testnet STX from the faucet for development

You understand:
- Stacks blockchain architecture and Proof of Transfer (PoX)
- Clarity programming language
- STX token economics and stacking
- Bitcoin integration and Bitcoin DeFi protocols
- Popular Stacks DeFi protocols (Alex, Arkadiko, etc.)
- Transaction fees and optimization strategies

Always provide accurate, actionable advice for Stacks blockchain operations. Use your tools when real-time blockchain data is needed."""

    def process_transaction_request(self, request: str) -> str:
        """
        Process a transaction request using AI analysis.
        
        Args:
            request: The transaction request or blockchain query
            
        Returns:
            AI-generated response with transaction guidance
        """
        # Create messages list with system message and transaction history
        messages = [SystemMessage(content=self.system_message)]
        
        # Add transaction history for context
        for entry in self.transaction_history[-5:]:  # Last 5 transactions for context
            messages.append(HumanMessage(content=f"Previous request: {entry['request']}"))
            messages.append(AIMessage(content=f"Previous response: {entry['response']}"))
        
        # Add current request
        messages.append(HumanMessage(content=request))
        
        # Get response from the model
        response = self.llm.invoke(messages)
        agent_response = response.content
        
        # Extract token usage
        user_tokens = 0
        response_tokens = 0
        
        # Try to get token usage from the response metadata
        if hasattr(response, 'response_metadata') and response.response_metadata:
            metadata = response.response_metadata
            if 'token_usage' in metadata:
                token_usage = metadata['token_usage']
                user_tokens = token_usage.get('input_tokens', 0)
                response_tokens = token_usage.get('output_tokens', 0)
            elif 'usage' in metadata:
                usage = metadata['usage']
                user_tokens = usage.get('input_tokens', 0)
                response_tokens = usage.get('output_tokens', 0)
        
        # Try alternative attribute names
        if hasattr(response, 'usage_metadata') and response.usage_metadata:
            usage = response.usage_metadata
            user_tokens = usage.get('input_tokens', user_tokens)
            response_tokens = usage.get('output_tokens', response_tokens)
        
        # Try direct attributes
        if hasattr(response, 'input_tokens'):
            user_tokens = response.input_tokens
        if hasattr(response, 'output_tokens'):
            response_tokens = response.output_tokens
        
        # If still no tokens found, try to estimate from content length
        if user_tokens == 0 and response_tokens == 0:
            # Rough estimation: ~4 characters per token
            estimated_user_tokens = len(request) // 4
            estimated_response_tokens = len(agent_response) // 4
            user_tokens = max(1, estimated_user_tokens)  # At least 1 token
            response_tokens = max(1, estimated_response_tokens)  # At least 1 token
        
        # Log token usage
        self._log_token_usage(user_tokens, response_tokens, self.sap_api_key)
        
        # Store in transaction history
        self.transaction_history.append({
            "request": request,
            "response": agent_response,
            "timestamp": self._get_timestamp(),
            "user_tokens": user_tokens,
            "response_tokens": response_tokens
        })
        
        return agent_response
    
    def create_transaction_chain(self, prompt_template: str) -> LLMChain:
        """
        Create a specialized LangChain chain for transaction processing.
        
        Args:
            prompt_template: The prompt template string with variables in {brackets}
            
        Returns:
            A LangChain LLMChain object optimized for Stacks operations
        """
        prompt = PromptTemplate.from_template(prompt_template)
        return LLMChain(llm=self.llm, prompt=prompt)
    
    def execute_chain(self, chain: LLMChain, **kwargs) -> str:
        """
        Execute a transaction chain with the provided variables.
        
        Args:
            chain: The LangChain chain to execute
            **kwargs: Variables to pass to the chain
            
        Returns:
            The chain's output as a string
        """
        return chain.run(**kwargs)
    
    def analyze_smart_contract(self, contract_code: str, analysis_type: str = "security") -> str:
        """
        Analyze a Clarity smart contract using AI.
        
        Args:
            contract_code: The Clarity smart contract code
            analysis_type: Type of analysis (security, optimization, functionality)
            
        Returns:
            AI analysis of the smart contract
        """
        analysis_prompt = f"""
        Analyze this Clarity smart contract for {analysis_type}:
        
        Contract Code:
        {contract_code}
        
        Please provide a detailed {analysis_type} analysis including:
        - Potential issues or vulnerabilities
        - Optimization suggestions
        - Best practices recommendations
        - Gas efficiency considerations
        """
        
        return self.process_transaction_request(analysis_prompt)
    
    def generate_clarity_code(self, description: str) -> str:
        """
        Generate Clarity smart contract code based on description.
        
        Args:
            description: Description of the desired smart contract functionality
            
        Returns:
            Generated Clarity code with explanations
        """
        code_prompt = f"""
        Generate a Clarity smart contract based on this description:
        {description}
        
        Please provide:
        1. Complete Clarity code
        2. Explanation of key functions
        3. Deployment considerations
        4. Testing recommendations
        """
        
        return self.process_transaction_request(code_prompt)
    
    def get_defi_strategy(self, strategy_type: str, amount: float = None) -> str:
        """
        Get DeFi strategy recommendations for Stacks ecosystem.
        
        Args:
            strategy_type: Type of DeFi strategy (staking, lending, yield-farming, etc.)
            amount: Optional amount of STX to consider
            
        Returns:
            DeFi strategy recommendations
        """
        amount_text = f" with {amount} STX" if amount else ""
        strategy_prompt = f"""
        Provide a {strategy_type} strategy for the Stacks DeFi ecosystem{amount_text}.
        
        Include:
        - Recommended protocols and platforms
        - Risk assessment
        - Expected returns and timeframes
        - Step-by-step execution plan
        - Gas fee considerations
        """
        
        return self.process_transaction_request(strategy_prompt)
    
    def get_account_balance(self, principal: str, network: str = "testnet") -> Dict[str, Any]:
        """
        Get the balance of a Stacks account using the Hiro API.
        
        Args:
            principal: The Stacks address (wallet address) to check
            network: Network to query ('mainnet' or 'testnet')
            
        Returns:
            Dictionary containing balance information including STX and tokens
        """
        # Strip whitespace from principal
        principal = principal.strip()
        
        # Determine the base URL based on network
        if network == "mainnet":
            base_url = "https://api.hiro.so/extended"
        else:  # testnet
            base_url = "https://api.testnet.hiro.so/extended"
        
        # Construct the API endpoint
        url = f"{base_url}/v1/address/{principal}/balances"
        
        headers = {
            'Accept': 'application/json'
        }
        
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()  # Raises an HTTPError for bad responses
            
            balance_data = response.json()
            
            # Process and format the balance data
            formatted_balance = {
                "address": principal,
                "network": network,
                "stx_balance": {
                    "balance": int(balance_data.get("stx", {}).get("balance", 0)),
                    "balance_stx": float(balance_data.get("stx", {}).get("balance", 0)) / 1000000,  # Convert microSTX to STX
                    "locked": int(balance_data.get("stx", {}).get("locked", 0)),
                    "locked_stx": float(balance_data.get("stx", {}).get("locked", 0)) / 1000000,
                    "unlock_height": balance_data.get("stx", {}).get("unlock_height", 0),
                    "burnchain_lock_height": balance_data.get("stx", {}).get("burnchain_lock_height", 0),
                    "burnchain_unlock_height": balance_data.get("stx", {}).get("burnchain_unlock_height", 0)
                },
                "fungible_tokens": balance_data.get("fungible_tokens", {}),
                "non_fungible_tokens": balance_data.get("non_fungible_tokens", {})
            }
            
            return formatted_balance
            
        except requests.exceptions.RequestException as e:
            error_msg = f"Error fetching balance for {principal}: {str(e)}"
            return {
                "error": error_msg,
                "address": principal,
                "network": network
            }
    
    def analyze_account_balance(self, principal: str, network: str = "testnet") -> str:
        """
        Get account balance and provide AI analysis of the wallet status using tools.
        
        Args:
            principal: The Stacks address to analyze
            network: Network to query ('mainnet' or 'testnet')
            
        Returns:
            AI analysis of the account balance and recommendations
        """
        # Strip whitespace from principal
        principal = principal.strip()
        
        # Use the tool to get balance data
        balance_tool = AccountBalanceTool()
        balance_data = balance_tool._run(principal, network)
        
        if "error" in balance_data:
            return f"Unable to fetch balance: {balance_data['error']}"
        
        analysis_prompt = f"""
        Analyze this Stacks wallet balance and provide insights:
        
        Address: {balance_data['address']}
        Network: {balance_data['network']}
        
        STX Balance: {balance_data['stx_balance']['balance_stx']:.6f} STX
        Locked STX: {balance_data['stx_balance']['locked_stx']:.6f} STX
        Unlock Height: {balance_data['stx_balance']['unlock_height']}
        
        Fungible Tokens: {len(balance_data['fungible_tokens'])} different tokens
        NFTs: {len(balance_data['non_fungible_tokens'])} different collections
        
        Please provide:
        1. Wallet health assessment
        2. Stacking status analysis (if STX are locked)
        3. Token portfolio overview
        4. Recommendations for optimization
        5. Potential DeFi opportunities
        """
        
        return self.process_transaction_request(analysis_prompt)
    
    def get_account_assets(self, principal: str, network: str = "testnet", limit: int = 50, offset: int = 0, 
                          unanchored: bool = True, until_block: str = None) -> Dict[str, Any]:
        """
        Get all assets (fungible and non-fungible tokens) held by an account.
        
        Args:
            principal: The Stacks address to query
            network: Network to query ('mainnet' or 'testnet')
            limit: Maximum number of assets to return (default: 50)
            offset: Number of assets to skip (default: 0)
            unanchored: Include unanchored transactions (default: True)
            until_block: Query up to specific block (optional)
            
        Returns:
            Dictionary containing asset information
        """
        # Strip whitespace from principal
        principal = principal.strip()
        
        # Determine the base URL based on network
        if network == "mainnet":
            base_url = "https://api.hiro.so/extended"
        else:  # testnet
            base_url = "https://api.testnet.hiro.so/extended"
        
        # Construct the API endpoint
        url = f"{base_url}/v1/address/{principal}/assets"
        
        # Build query parameters
        params = {
            "limit": limit,
            "offset": offset,
            "unanchored": str(unanchored).lower()
        }
        if until_block:
            params["until_block"] = until_block
        
        headers = {'Accept': 'application/json'}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            assets_data = response.json()
            
            return {
                "address": principal,
                "network": network,
                "total_assets": assets_data.get("total", 0),
                "limit": limit,
                "offset": offset,
                "results": assets_data.get("results", [])
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Error fetching assets for {principal}: {str(e)}",
                "address": principal,
                "network": network
            }
    
    def get_stx_inbound(self, principal: str, network: str = "testnet", limit: int = 50, offset: int = 0,
                       height: int = None, unanchored: bool = True, until_block: str = None) -> Dict[str, Any]:
        """
        Get inbound STX transfers for an account.
        
        Args:
            principal: The Stacks address to query
            network: Network to query ('mainnet' or 'testnet')
            limit: Maximum number of transactions to return (default: 50)
            offset: Number of transactions to skip (default: 0)
            height: Filter by block height (optional)
            unanchored: Include unanchored transactions (default: True)
            until_block: Query up to specific block (optional)
            
        Returns:
            Dictionary containing inbound STX transfer data
        """
        # Strip whitespace from principal
        principal = principal.strip()
        
        # Determine the base URL based on network
        if network == "mainnet":
            base_url = "https://api.hiro.so/extended"
        else:  # testnet
            base_url = "https://api.testnet.hiro.so/extended"
        
        # Construct the API endpoint
        url = f"{base_url}/v1/address/{principal}/stx_inbound"
        
        # Build query parameters
        params = {
            "limit": limit,
            "offset": offset,
            "unanchored": str(unanchored).lower()
        }
        if height:
            params["height"] = height
        if until_block:
            params["until_block"] = until_block
        
        headers = {'Accept': 'application/json'}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            inbound_data = response.json()
            
            return {
                "address": principal,
                "network": network,
                "total_inbound": inbound_data.get("total", 0),
                "limit": limit,
                "offset": offset,
                "results": inbound_data.get("results", [])
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Error fetching inbound STX for {principal}: {str(e)}",
                "address": principal,
                "network": network
            }
    
    def get_account_transactions(self, principal: str, network: str = "testnet", limit: int = 50, offset: int = 0,
                               height: int = None, unanchored: bool = True, until_block: str = None,
                               exclude_function_args: bool = False) -> Dict[str, Any]:
        """
        Get transaction history for an account.
        
        Args:
            principal: The Stacks address to query
            network: Network to query ('mainnet' or 'testnet')
            limit: Maximum number of transactions to return (default: 50)
            offset: Number of transactions to skip (default: 0)
            height: Filter by block height (optional)
            unanchored: Include unanchored transactions (default: True)
            until_block: Query up to specific block (optional)
            exclude_function_args: Exclude function arguments from results (default: False)
            
        Returns:
            Dictionary containing transaction history
        """
        # Strip whitespace from principal
        principal = principal.strip()
        
        # Determine the base URL based on network
        if network == "mainnet":
            base_url = "https://api.hiro.so/extended"
        else:  # testnet
            base_url = "https://api.testnet.hiro.so/extended"
        
        # Construct the API endpoint
        url = f"{base_url}/v1/address/{principal}/transactions"
        
        # Build query parameters
        params = {
            "limit": limit,
            "offset": offset,
            "unanchored": str(unanchored).lower(),
            "exclude_function_args": str(exclude_function_args).lower()
        }
        if height:
            params["height"] = height
        if until_block:
            params["until_block"] = until_block
        
        headers = {'Accept': 'application/json'}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            tx_data = response.json()
            
            return {
                "address": principal,
                "network": network,
                "total_transactions": tx_data.get("total", 0),
                "limit": limit,
                "offset": offset,
                "results": tx_data.get("results", [])
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Error fetching transactions for {principal}: {str(e)}",
                "address": principal,
                "network": network
            }
    
    def get_account_nonces(self, principal: str, network: str = "testnet", block_height: int = None, 
                          block_hash: str = None) -> Dict[str, Any]:
        """
        Get nonce information for an account at a specific block.
        
        Args:
            principal: The Stacks address to query
            network: Network to query ('mainnet' or 'testnet')
            block_height: Block height to query (optional)
            block_hash: Block hash to query (optional)
            
        Returns:
            Dictionary containing nonce information
        """
        # Strip whitespace from principal
        principal = principal.strip()
        
        # Determine the base URL based on network
        if network == "mainnet":
            base_url = "https://api.hiro.so/extended"
        else:  # testnet
            base_url = "https://api.testnet.hiro.so/extended"
        
        # Construct the API endpoint
        url = f"{base_url}/v1/address/{principal}/nonces"
        
        # Build query parameters
        params = {}
        if block_height:
            params["block_height"] = block_height
        if block_hash:
            params["block_hash"] = block_hash
        
        headers = {'Accept': 'application/json'}
        
        try:
            response = requests.get(url, headers=headers, params=params, timeout=10)
            response.raise_for_status()
            
            nonce_data = response.json()
            
            return {
                "address": principal,
                "network": network,
                "nonce_data": nonce_data
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Error fetching nonces for {principal}: {str(e)}",
                "address": principal,
                "network": network
            }
    
    def request_testnet_stx(self, address: str, stacking: bool = False) -> Dict[str, Any]:
        """
        Request testnet STX from the faucet (testnet only).
        
        Args:
            address: The Stacks address to send testnet STX to
            stacking: Whether the address will be used for stacking (default: False)
            
        Returns:
            Dictionary containing faucet response
        """
        # This endpoint only works on testnet
        base_url = "https://api.testnet.hiro.so/extended"
        url = f"{base_url}/v1/faucets/stx"
        
        # Build query parameters
        params = {
            "address": address,
            "stacking": str(stacking).lower()
        }
        
        headers = {'Accept': 'application/json'}
        
        try:
            response = requests.post(url, headers=headers, params=params, timeout=15)
            response.raise_for_status()
            
            faucet_data = response.json()
            
            return {
                "address": address,
                "stacking": stacking,
                "success": True,
                "response": faucet_data
            }
            
        except requests.exceptions.RequestException as e:
            return {
                "error": f"Error requesting testnet STX for {address}: {str(e)}",
                "address": address,
                "stacking": stacking,
                "success": False
            }
    
    def analyze_account_activity(self, principal: str, network: str = "testnet", limit: int = 20) -> str:
        """
        Get comprehensive account activity analysis using multiple tools.
        
        Args:
            principal: The Stacks address to analyze
            network: Network to query ('mainnet' or 'testnet')
            limit: Number of recent transactions to analyze
            
        Returns:
            AI analysis of account activity
        """
        # Strip whitespace from principal
        principal = principal.strip()
        
        # Use tools to gather data from multiple endpoints
        balance_tool = AccountBalanceTool()
        assets_tool = AccountAssetsTool()
        transactions_tool = AccountTransactionsTool()
        inbound_tool = STXInboundTool()
        
        balance = balance_tool._run(principal, network)
        assets = assets_tool._run(principal, network, limit=20)
        transactions = transactions_tool._run(principal, network, limit=limit)
        inbound = inbound_tool._run(principal, network, limit=10)
        
        # Check for errors
        errors = []
        if "error" in balance:
            errors.append(f"Balance: {balance['error']}")
        if "error" in assets:
            errors.append(f"Assets: {assets['error']}")
        if "error" in transactions:
            errors.append(f"Transactions: {transactions['error']}")
        if "error" in inbound:
            errors.append(f"Inbound: {inbound['error']}")
        
        if errors:
            return f"Unable to fetch complete account data: {'; '.join(errors)}"
        
        analysis_prompt = f"""
        Provide a comprehensive analysis of this Stacks account activity:
        
        Address: {principal}
        Network: {network}
        
        BALANCE INFORMATION:
        - STX Balance: {balance.get('stx_balance', {}).get('balance_stx', 0):.6f} STX
        - Locked STX: {balance.get('stx_balance', {}).get('locked_stx', 0):.6f} STX
        - Fungible Tokens: {len(balance.get('fungible_tokens', {}))} types
        - NFT Collections: {len(balance.get('non_fungible_tokens', {}))} types
        
        ASSET HOLDINGS:
        - Total Assets: {assets.get('total_assets', 0)}
        - Recent Assets: {len(assets.get('results', []))} shown
        
        TRANSACTION ACTIVITY:
        - Total Transactions: {transactions.get('total_transactions', 0)}
        - Recent Transactions: {len(transactions.get('results', []))} shown
        
        INBOUND STX TRANSFERS:
        - Total Inbound: {inbound.get('total_inbound', 0)}
        - Recent Inbound: {len(inbound.get('results', []))} shown
        
        Please provide:
        1. Account activity level assessment (active/inactive/moderate)
        2. Transaction pattern analysis
        3. Asset portfolio overview
        4. Stacking participation analysis
        5. Recommendations for account optimization
        6. Potential security considerations
        7. DeFi engagement opportunities
        """
        
        return self.process_transaction_request(analysis_prompt)
    
    def update_blockchain_context(self, context_updates: Dict[str, Any]):
        """
        Update the blockchain context information.
        
        Args:
            context_updates: Dictionary of context updates
        """
        self.blockchain_context.update(context_updates)
    
    def clear_transaction_history(self):
        """Clear the transaction history."""
        self.transaction_history = []
    
    def get_transaction_history(self) -> List[Dict]:
        """
        Get the transaction history.
        
        Returns:
            List of transaction entries
        """
        return self.transaction_history.copy()
    
    def set_system_message(self, message: str):
        """
        Update the system message.
        
        Args:
            message: New system message for the agent
        """
        self.system_message = message
        # Clear history since system message changed
        self.clear_transaction_history()
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for transaction history."""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def get_tools(self) -> List[BaseTool]:
        """
        Get all available tools for the agent.
        
        Returns:
            List of BaseTool instances
        """
        return self.tools.copy()
    
    def close_connections(self):
        """Close MongoDB connection"""
        if self.mongodb_client:
            self.mongodb_client.close()
            self.mongodb_client = None
            self.api_keys_collection = None
    
    def __del__(self):
        """Cleanup when object is destroyed"""
        self.close_connections()
    
    def create_agent_executor(self) -> AgentExecutor:
        """
        Create an AgentExecutor with tools for advanced agent capabilities.
        
        Returns:
            An AgentExecutor instance with access to all Stacks blockchain tools
        """
        from langchain.agents import create_react_agent
        from langchain import hub
        
        try:
            # Try to get the standard ReAct prompt from hub
            prompt = hub.pull("hwchase17/react")
        except:
            # Fallback to a custom ReAct prompt
            from langchain.prompts import PromptTemplate
            prompt = PromptTemplate.from_template("""
You are a helpful assistant that can use tools to answer questions about the Stacks blockchain.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: {input}
Thought:{agent_scratchpad}
""")
        
        # Create the agent using ReAct pattern
        agent = create_react_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create the executor
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            verbose=True,
            return_intermediate_steps=True
        )
    
    def process_with_tools(self, request: str) -> str:
        """
        Process a request using the agent with tools.
        
        Args:
            request: The request or query to process
            
        Returns:
            The agent's response after potentially using tools
        """
        try:
            # Try the agent executor first
            agent_executor = self.create_agent_executor()
            result = agent_executor.invoke({"input": request})
            response = result["output"]
            
            # Extract token usage from the result
            user_tokens = 0
            response_tokens = 0
            
            # Try to get token usage from the LLM response
            if hasattr(result, 'llm_output') and result.get('llm_output'):
                llm_output = result['llm_output']
                if 'token_usage' in llm_output:
                    token_usage = llm_output['token_usage']
                    user_tokens = token_usage.get('input_tokens', 0)
                    response_tokens = token_usage.get('output_tokens', 0)
                elif 'usage' in llm_output:
                    usage = llm_output['usage']
                    user_tokens = usage.get('input_tokens', 0)
                    response_tokens = usage.get('output_tokens', 0)
            
            # If still no tokens found, estimate from request and response
            if user_tokens == 0 and response_tokens == 0:
                estimated_user_tokens = len(request) // 4
                estimated_response_tokens = len(response) // 4
                user_tokens = max(1, estimated_user_tokens)
                response_tokens = max(1, estimated_response_tokens)
            
            # Log token usage
            self._log_token_usage(user_tokens, response_tokens, self.sap_api_key)
            
            # Store in transaction history
            self.transaction_history.append({
                "request": request,
                "response": response,
                "timestamp": self._get_timestamp(),
                "used_tools": True,
                "user_tokens": user_tokens,
                "response_tokens": response_tokens
            })
            
            return response
            
        except Exception as e:
            # Fallback to simple tool-based processing
            return self._process_with_tools_fallback(request, str(e))
    
    def _process_with_tools_fallback(self, request: str, error_msg: str) -> str:
        """
        Fallback method for processing requests with tools when agent executor fails.
        
        Args:
            request: The request or query to process
            error_msg: The error message from the failed agent executor
            
        Returns:
            The agent's response using direct tool calls
        """
        # Enhanced prompt that includes tool information
        enhanced_prompt = f"""
        {self.system_message}
        
        You have access to the following tools for blockchain interactions:
        - get_account_balance: Get STX and token balances for any Stacks address
        - get_account_assets: Get all assets held by a Stacks address
        - get_stx_inbound: Get inbound STX transfers for an address
        - get_account_transactions: Get transaction history for an address
        - get_account_nonces: Get nonce information for an address
        - request_testnet_stx: Request testnet STX from the faucet
        
        User Request: {request}
        
        Please provide a helpful response. If the request involves checking blockchain data for a specific address, 
        I can help you understand what tools would be needed, but you'll need to call them directly.
        """
        
        try:
            # Use the regular processing method with enhanced context
            response = self.process_transaction_request(enhanced_prompt)
            
            # Add a note about tool availability
            tool_note = "\n\nNote: For real-time blockchain data, you can use the individual tools directly:"
            for tool in self.tools:
                tool_note += f"\n- {tool.name}: {tool.description}"
            
            response += tool_note
            
            # Store in transaction history
            self.transaction_history.append({
                "request": request,
                "response": response,
                "timestamp": self._get_timestamp(),
                "used_tools": False,
                "fallback": True,
                "original_error": error_msg
            })
            
            return response
            
        except Exception as fallback_error:
            error_response = f"Error processing request: {str(fallback_error)}"
            
            # Store error in transaction history
            self.transaction_history.append({
                "request": request,
                "response": error_response,
                "timestamp": self._get_timestamp(),
                "used_tools": False,
                "error": True,
                "fallback_error": str(fallback_error),
                "original_error": error_msg
            })
            
            return error_response


# Convenience function for quick agent creation
def create_stacks_agent(
    model_name: str = "claude-3-5-sonnet-20240620",
    temperature: float = 0.7,
    system_message: Optional[str] = None,
    api_key: Optional[str] = None,
    sap_api_key: Optional[str] = None,
    validate_api_key: bool = True
) -> StacksAgentProtocol:
    """
    Convenience function to create a Stacks Agent Protocol instance.
    
    Args:
        model_name: The Claude model to use (e.g., claude-3-5-sonnet-20240620, claude-3-5-haiku-20240620)
        temperature: Temperature for response generation
        system_message: Optional system message
        api_key: Anthropic API key
        sap_api_key: SAP Agent Protocol API key for authentication
        validate_api_key: Whether to validate the SAP API key
        
    Returns:
        A StacksAgentProtocol instance
    """
    return StacksAgentProtocol(
        model_name=model_name,
        temperature=temperature,
        system_message=system_message,
        api_key=api_key,
        sap_api_key=sap_api_key,
        validate_api_key=validate_api_key
    )