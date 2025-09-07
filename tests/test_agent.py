#!/usr/bin/env python3
"""
Tests for Stacks Agent Protocol
"""

import pytest
import os
from unittest.mock import Mock, patch
from stacks_agent_protocol import create_stacks_agent, StacksAgentProtocol

class TestStacksAgentProtocol:
    """Test cases for StacksAgentProtocol class"""
    
    def test_agent_initialization(self):
        """Test agent initialization with valid parameters"""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            agent = create_stacks_agent(
                sap_api_key="test-sap-key",
                anthropic_api_key="test-anthropic-key",
                validate_api_key=False
            )
            assert agent is not None
            assert agent.sap_api_key == "test-sap-key"
            agent.close_connections()
    
    def test_agent_without_api_key(self):
        """Test agent initialization without API key"""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="Anthropic API key must be provided"):
                create_stacks_agent(
                    sap_api_key="test-sap-key",
                    validate_api_key=False
                )
    
    def test_agent_with_environment_key(self):
        """Test agent initialization with environment API key"""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'env-key'}):
            agent = create_stacks_agent(
                sap_api_key="test-sap-key",
                validate_api_key=False
            )
            assert agent is not None
            agent.close_connections()
    
    @patch('stacks_agent_protocol.MongoClient')
    def test_mongodb_connection(self, mock_mongo):
        """Test MongoDB connection initialization"""
        mock_client = Mock()
        mock_mongo.return_value = mock_client
        
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            agent = create_stacks_agent(
                sap_api_key="test-sap-key",
                validate_api_key=False
            )
            assert agent.mongodb_client is not None
            agent.close_connections()
    
    def test_transaction_history(self):
        """Test transaction history functionality"""
        with patch.dict(os.environ, {'ANTHROPIC_API_KEY': 'test-key'}):
            agent = create_stacks_agent(
                sap_api_key="test-sap-key",
                validate_api_key=False
            )
            
            # Initially empty
            history = agent.get_transaction_history()
            assert isinstance(history, list)
            assert len(history) == 0
            
            agent.close_connections()

if __name__ == "__main__":
    pytest.main([__file__])