.PHONY: help install install-dev test lint format clean build upload

help: ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install: ## Install the package
	pip install -e .

install-dev: ## Install development dependencies
	pip install -e ".[dev]"

test: ## Run tests
	pytest tests/ -v

lint: ## Run linting
	flake8 stacks_agent_protocol/ tests/
	mypy stacks_agent_protocol/

format: ## Format code
	black stacks_agent_protocol/ tests/ example.py

clean: ## Clean build artifacts
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete

build: clean ## Build the package
	python -m build

upload: build ## Upload to PyPI
	twine upload dist/*

upload-test: build ## Upload to Test PyPI
	twine upload --repository testpypi dist/*

check: ## Check package
	twine check dist/*

example: ## Run example script
	python example.py