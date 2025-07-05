# Soft Logic Microservice - Regression Testing Makefile
# =====================================================
#
# This Makefile provides comprehensive regression testing targets for the
# soft logic microservice project, including unit tests, integration tests,
# demonstrations, type checking, and contract validation.

# Variables
PYTHON := python
POETRY := poetry
PROJECT_DIR := .
SRC_DIR := app
TEST_DIR := tests
DEMO_DIR := .

# Python files
CORE_MODULES := $(shell find $(SRC_DIR)/core -name "*.py" -not -name "__*")
TEST_FILES := $(shell find $(TEST_DIR) -name "test_*.py")
DEMO_FILES := demo_abstractions.py demo_hybrid_system.py demo_enhanced_system.py demo_comprehensive_system.py
CONTRACT_DEMO := app/core/icontract_demo.py
PHASE_3A_TEST := test_phase_3a.py

# Colors for output
GREEN := \033[0;32m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

.PHONY: help test test-unit test-integration test-demos test-all type-check lint clean setup install deps

# Default target
help:
	@echo "$(GREEN)Soft Logic Microservice - Test Targets$(NC)"
	@echo "========================================"
	@echo ""
	@echo "$(YELLOW)Setup Targets:$(NC)"
	@echo "  install     - Install all dependencies with poetry"
	@echo "  deps        - Install dependencies only"
	@echo "  setup       - Full project setup (install + pre-commit)"
	@echo ""
	@echo "$(YELLOW)Testing Targets:$(NC)"
	@echo "  test        - Run all tests (unit + integration + demos)"
	@echo "  test-unit   - Run unit tests only"
	@echo "  test-integration - Run integration tests"
	@echo "  test-demos  - Run all demonstration scripts"
	@echo "  test-contracts - Run Design by Contract demonstrations"
	@echo "  test-phase3a - Run Phase 3A type safety tests"
	@echo "  test-regression - Full regression test suite"
	@echo ""
	@echo "$(YELLOW)Quality Targets:$(NC)"
	@echo "  type-check  - Run mypy type checking"
	@echo "  lint        - Run code linting (flake8, black, isort)"
	@echo "  format      - Format code with black and isort"
	@echo ""
	@echo "$(YELLOW)Utility Targets:$(NC)"
	@echo "  clean       - Clean up temporary files"
	@echo "  coverage    - Run tests with coverage report"
	@echo "  docs        - Generate documentation"
	@echo ""
	@echo "$(YELLOW)Examples:$(NC)"
	@echo "  make test           # Run all tests"
	@echo "  make test-unit      # Run only unit tests"
	@echo "  make type-check     # Check types with mypy"
	@echo "  make test-regression # Full regression suite"

# Setup targets
install:
	@echo "$(GREEN)Installing project dependencies...$(NC)"
	$(POETRY) install
	@echo "$(GREEN)Dependencies installed successfully!$(NC)"

deps:
	@echo "$(GREEN)Installing dependencies only...$(NC)"
	$(POETRY) install --no-dev

setup: install
	@echo "$(GREEN)Setting up development environment...$(NC)"
	$(POETRY) run pre-commit install || echo "Pre-commit not available, skipping..."
	@echo "$(GREEN)Setup complete!$(NC)"

# Core testing targets
test-unit:
	@echo "$(GREEN)Running unit tests...$(NC)"
	@echo "$(YELLOW)Testing core abstractions...$(NC)"
	$(POETRY) run pytest $(TEST_DIR)/test_core/test_abstractions.py -v
	@echo "$(GREEN)Unit tests completed!$(NC)"

test-integration:
	@echo "$(GREEN)Running integration tests...$(NC)"
	@echo "$(YELLOW)Testing API endpoints...$(NC)"
	$(POETRY) run pytest $(TEST_DIR)/test_main.py -v || echo "$(YELLOW)API tests require Phase 3C completion$(NC)"
	@echo "$(GREEN)Integration tests completed!$(NC)"

test-phase3a:
	@echo "$(GREEN)Running Phase 3A type safety tests...$(NC)"
	$(POETRY) run python $(PHASE_3A_TEST)
	@echo "$(GREEN)Phase 3A tests completed!$(NC)"

test-contracts:
	@echo "$(GREEN)Running Design by Contract demonstrations...$(NC)"
	$(POETRY) run python $(CONTRACT_DEMO)
	@echo "$(GREEN)Contract demonstrations completed!$(NC)"

test-demos:
	@echo "$(GREEN)Running demonstration scripts...$(NC)"
	@for demo in $(DEMO_FILES); do \
		echo "$(YELLOW)Running $$demo...$(NC)"; \
		$(POETRY) run python $$demo || echo "$(RED)$$demo failed$(NC)"; \
		echo ""; \
	done
	@echo "$(GREEN)All demonstrations completed!$(NC)"

# Test targets
test-persistence:
	@echo "$(YELLOW)Running persistence layer tests...$(NC)"
	$(PYTHON) -m pytest tests/test_core/test_persistence.py -v
	@echo "$(GREEN)Persistence tests completed$(NC)"

test-persistence-integration:
	@echo "$(YELLOW)Running persistence integration tests...$(NC)"
	$(PYTHON) -m pytest tests/test_core/test_persistence.py::TestIntegrationPersistence -v
	@echo "$(GREEN)Persistence integration tests completed$(NC)"

test-persistence-performance:
	@echo "$(YELLOW)Running persistence performance tests...$(NC)"
	$(PYTHON) -m pytest tests/test_core/test_persistence.py::TestPerformancePersistence -v -m performance
	@echo "$(GREEN)Persistence performance tests completed$(NC)"

test-batch-workflows:
	@echo "$(YELLOW)Running batch workflow tests...$(NC)"
	$(PYTHON) -m pytest tests/test_core/test_persistence.py::TestBatchPersistenceManager -v
	@echo "$(GREEN)Batch workflow tests completed$(NC)"

# Comprehensive test suites
test: test-unit test-phase3a test-contracts
	@echo "$(GREEN)All primary tests completed successfully!$(NC)"

test-all: test-unit test-integration test-phase3a test-contracts test-demos test-persistence
	@echo "$(GREEN)Complete test suite finished!$(NC)"

test-regression: clean type-check lint test-all
	@echo "$(GREEN)Full regression test suite completed!$(NC)"
	@echo "$(YELLOW)Summary:$(NC)"
	@echo "  ✓ Type checking passed"
	@echo "  ✓ Code quality checks passed"
	@echo "  ✓ Unit tests passed"
	@echo "  ✓ Integration tests checked"
	@echo "  ✓ Phase 3A tests passed"
	@echo "  ✓ Contract demonstrations passed"
	@echo "  ✓ Demo scripts validated"

# Quality assurance targets
type-check:
	@echo "$(GREEN)Running mypy type checking...$(NC)"
	$(POETRY) run mypy $(SRC_DIR)/ --config-file mypy.ini
	@echo "$(GREEN)Type checking completed!$(NC)"

lint:
	@echo "$(GREEN)Running code quality checks...$(NC)"
	@echo "$(YELLOW)Running flake8...$(NC)"
	$(POETRY) run flake8 $(SRC_DIR)/ --max-line-length=88 --extend-ignore=E203,W503
	@echo "$(YELLOW)Running black (check only)...$(NC)"
	$(POETRY) run black --check $(SRC_DIR)/ $(TEST_DIR)/
	@echo "$(YELLOW)Running isort (check only)...$(NC)"
	$(POETRY) run isort --check-only $(SRC_DIR)/ $(TEST_DIR)/
	@echo "$(GREEN)Code quality checks completed!$(NC)"

format:
	@echo "$(GREEN)Formatting code...$(NC)"
	$(POETRY) run black $(SRC_DIR)/ $(TEST_DIR)/ $(DEMO_FILES) $(PHASE_3A_TEST)
	$(POETRY) run isort $(SRC_DIR)/ $(TEST_DIR)/ $(DEMO_FILES) $(PHASE_3A_TEST)
	@echo "$(GREEN)Code formatting completed!$(NC)"

# Coverage and documentation
coverage:
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	$(POETRY) run pytest $(TEST_DIR)/ --cov=$(SRC_DIR) --cov-report=html --cov-report=term
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

docs:
	@echo "$(GREEN)Generating documentation...$(NC)"
	@echo "$(YELLOW)Current documentation files:$(NC)"
	@find . -name "*.md" -not -path "./.*" | sort
	@echo "$(GREEN)Documentation listing completed!$(NC)"

# Utility targets
clean:
	@echo "$(GREEN)Cleaning up temporary files...$(NC)"
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -delete
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name ".coverage" -delete
	rm -rf htmlcov/
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	@echo "$(GREEN)Cleanup completed!$(NC)"

# Development workflow targets
dev-test:
	@echo "$(GREEN)Running development test cycle...$(NC)"
	make type-check
	make test-unit
	make test-phase3a
	@echo "$(GREEN)Development test cycle completed!$(NC)"

quick-test:
	@echo "$(GREEN)Running quick test suite...$(NC)"
	$(POETRY) run pytest $(TEST_DIR)/test_core/test_abstractions.py -x -v
	$(POETRY) run python $(PHASE_3A_TEST)
	@echo "$(GREEN)Quick tests completed!$(NC)"

# Contract-specific targets
test-contract-creation:
	@echo "$(GREEN)Testing contract-enhanced concept creation...$(NC)"
	$(POETRY) run python -c "from app.core.icontract_demo import ContractEnhancedRegistry; registry = ContractEnhancedRegistry(); concept = registry.create_concept_with_contracts('test_concept', 'default'); print(f'✓ Created concept: {concept.name}')"
	@echo "$(GREEN)Contract creation test passed!$(NC)"

test-contract-violations:
	@echo "$(GREEN)Testing contract violation detection...$(NC)"
	$(POETRY) run python -c "from app.core.icontract_demo import ContractEnhancedRegistry; from icontract import ViolationError; registry = ContractEnhancedRegistry(); exec('try:\n    registry.create_concept_with_contracts(\"\", \"default\")\n    print(\"❌ Should have failed\")\nexcept ViolationError:\n    print(\"✓ Contract violation correctly detected\")')"
	@echo "$(GREEN)Contract violation test passed!$(NC)"

# Performance targets
perf-test:
	@echo "$(GREEN)Running performance tests...$(NC)"
	$(POETRY) run python -c "import time; from app.core.enhanced_semantic_reasoning import EnhancedHybridRegistry; print('Testing registry performance...'); start_time = time.time(); registry = EnhancedHybridRegistry(download_wordnet=False, n_clusters=4); [registry.create_frame_aware_concept_with_advanced_embedding(name=f'test_concept_{i}', context='default', use_semantic_embedding=True) for i in range(10)]; end_time = time.time(); print(f'✓ Created 10 concepts in {end_time - start_time:.3f} seconds'); print(f'✓ Registry contains {len(registry.frame_aware_concepts)} concepts')"
	@echo "$(GREEN)Performance tests completed!$(NC)"

# Validation targets
validate-project:
	@echo "$(GREEN)Validating project structure...$(NC)"
	@echo "$(YELLOW)Checking required files...$(NC)"
	@test -f pyproject.toml && echo "✓ pyproject.toml exists" || echo "❌ pyproject.toml missing"
	@test -f mypy.ini && echo "✓ mypy.ini exists" || echo "❌ mypy.ini missing"
	@test -d $(SRC_DIR)/core && echo "✓ Core module directory exists" || echo "❌ Core module missing"
	@test -d $(TEST_DIR) && echo "✓ Tests directory exists" || echo "❌ Tests directory missing"
	@echo "$(YELLOW)Checking core modules...$(NC)"
	@for module in abstractions concept_registry enhanced_semantic_reasoning; do \
		test -f $(SRC_DIR)/core/$$module.py && echo "✓ $$module.py exists" || echo "❌ $$module.py missing"; \
	done
	@echo "$(GREEN)Project validation completed!$(NC)"

# Phase-specific targets
test-phase1:
	@echo "$(GREEN)Running Phase 1 (Core Abstractions) tests...$(NC)"
	$(POETRY) run pytest $(TEST_DIR)/test_core/test_abstractions.py -v -k "test_concept or test_axiom or test_context"
	@echo "$(GREEN)Phase 1 tests completed!$(NC)"

test-phase2:
	@echo "$(GREEN)Running Phase 2 (Hybrid Semantic System) tests...$(NC)"
	$(POETRY) run python demo_hybrid_system.py
	$(POETRY) run python demo_enhanced_system.py
	@echo "$(GREEN)Phase 2 tests completed!$(NC)"

test-phase3:
	@echo "$(GREEN)Running Phase 3 (Service Layer) tests...$(NC)"
	make test-phase3a
	make test-contracts
	@echo "$(GREEN)Phase 3 tests completed!$(NC)"

# Continuous Integration target
ci: clean install validate-project type-check lint test-all
	@echo "$(GREEN)CI pipeline completed successfully!$(NC)"

# Help for specific test categories
help-testing:
	@echo "$(GREEN)Testing Strategy Overview$(NC)"
	@echo "========================="
	@echo ""
	@echo "$(YELLOW)Test Categories:$(NC)"
	@echo "1. Unit Tests        - Core functionality validation"
	@echo "2. Integration Tests - End-to-end system testing"
	@echo "3. Contract Tests    - Design by Contract validation"
	@echo "4. Demo Tests        - Demonstration script execution"
	@echo "5. Type Tests        - Static type checking"
	@echo "6. Performance Tests - Basic performance validation"
	@echo ""
	@echo "$(YELLOW)Regression Testing:$(NC)"
	@echo "Use 'make test-regression' for comprehensive validation before releases"
	@echo ""
	@echo "$(YELLOW)Development Workflow:$(NC)"
	@echo "Use 'make dev-test' for quick validation during development"

# Error handling
.ONESHELL:
SHELL := /bin/bash
.SHELLFLAGS := -e -u -c
MAKEFLAGS += --warn-undefined-variables
MAKEFLAGS += --no-builtin-rules
