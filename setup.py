#!/usr/bin/env python3
"""
Setup script for Stacks Agent Protocol (SAP) package
"""

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return "Stacks Agent Protocol - A comprehensive agent for interacting with the Stacks blockchain"

# Read requirements
def read_requirements():
    requirements_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(requirements_path):
        with open(requirements_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return [
        'langchain>=0.1.0',
        'langchain-anthropic>=0.1.0',
        'pymongo>=4.0.0',
        'python-dotenv>=1.0.0',
        'requests>=2.28.0'
    ]

setup(
    name="stacks-agent-protocol",
    version="1.0.0",
    author="Stacks Agent Protocol Team",
    author_email="info@stacksagentprotocol.com",
    description="A comprehensive agent for interacting with the Stacks blockchain using LangChain and Claude AI",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/stacks-agent-protocol/stacks-agent-protocol",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Internet :: WWW/HTTP :: Dynamic Content",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    python_requires=">=3.8",
    install_requires=read_requirements(),
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-asyncio>=0.21.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=1.0.0",
        ],
        "streamlit": [
            "streamlit>=1.28.0",
        ],
    },
    keywords="stacks blockchain agent langchain claude ai cryptocurrency defi",
    project_urls={
        "Bug Reports": "https://github.com/stacks-agent-protocol/stacks-agent-protocol/issues",
        "Source": "https://github.com/stacks-agent-protocol/stacks-agent-protocol",
        "Documentation": "https://docs.stacksagentprotocol.com",
    },
    include_package_data=True,
    zip_safe=False,
)