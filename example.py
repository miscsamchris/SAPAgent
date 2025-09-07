#!/usr/bin/env python3
"""
Example usage of Stacks Agent Protocol
"""

import os
from stacks_agent_protocol import create_stacks_agent

def main():
    """Example usage of the Stacks Agent Protocol"""
    
    # Set up environment variables (in production, use .env file)
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY")
    sap_api_key = os.getenv("SAP_API_KEY")
    
    if not anthropic_api_key:
        print("Please set ANTHROPIC_API_KEY environment variable")
        return
    
    if not sap_api_key:
        print("Please set SAP_API_KEY environment variable")
        return
    
    try:
        # Create an agent instance
        print("Creating Stacks Agent Protocol instance...")
        agent = create_stacks_agent(
            sap_api_key=sap_api_key,
            anthropic_api_key=anthropic_api_key,
            validate_api_key=False  # Set to True in production
        )
        
        # Example 1: Check account balance
        print("\n=== Example 1: Checking Account Balance ===")
        response = agent.process_with_tools(
            "What is the balance of SP3K8BC0PPEVCV7NZ6QSRWPQ2JE9E5B6N3PA0KBR9?"
        )
        print(f"Response: {response}")
        
        # Example 2: Get account assets
        print("\n=== Example 2: Getting Account Assets ===")
        response = agent.process_with_tools(
            "Show me all assets for SP3K8BC0PPEVCV7NZ6QSRWPQ2JE9E5B6N3PA0KBR9"
        )
        print(f"Response: {response}")
        
        # Example 3: Check transaction history
        print("\n=== Example 3: Transaction History ===")
        response = agent.process_with_tools(
            "Show me the last 5 transactions for SP3K8BC0PPEVCV7NZ6QSRWPQ2JE9E5B6N3PA0KBR9"
        )
        print(f"Response: {response}")
        
        # Example 4: Get transaction history
        print("\n=== Example 4: Agent Transaction History ===")
        history = agent.get_transaction_history()
        print(f"Transaction History: {len(history)} entries")
        for i, entry in enumerate(history[-3:], 1):  # Show last 3 entries
            print(f"  {i}. {entry.get('request', 'N/A')[:50]}...")
        
        # Clean up
        agent.close_connections()
        print("\n✅ Examples completed successfully!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()