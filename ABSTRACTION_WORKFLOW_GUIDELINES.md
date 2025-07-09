# Abstraction Consistency and Data Model Standardization Workflow

## Purpose
This document provides a standardized workflow and set of guidelines for future LLM coding partners and new contributors to the ltnexp03 codebase. It ensures ongoing abstraction consistency, protocol/ABC usage, and data model standardization, supporting maintainability, type safety, and robust system evolution.

---

## 🛠️ Standardized Workflow for Abstraction Consistency

### 1. Work Incrementally and Test-Driven
- Make small, focused changes (e.g., standardize one model or one module at a time).
- After each change, run the full test suite (`pytest`), mypy, and icontract validation to catch regressions early.

### 2. Standardize Data Model Usage
- **API Boundaries:** Use Pydantic models for request/response validation.
- **Core Logic:** Use `@dataclass` for all internal data structures (e.g., `Concept`, `FrameInstance`).
- **Type Hints:** Use `TypedDict` only for type annotations, not for runtime objects.
- Remove duplicate or redundant model definitions (keep only the appropriate one for each context).
- Refactor code to use the standardized model in each context, updating imports and usages accordingly.

### 3. Add/Update Conversion Utilities
- Where models must be converted (e.g., dataclass ↔ Pydantic), add clear utility functions and test them.
- Ensure all conversions are covered by tests.

### 4. Update Documentation
- Update docstrings and markdown docs to reflect the new model usage patterns and conversion utilities.
- Document any deviations or exceptions in a dedicated section.

### 5. Maintain Contract and Type Safety
- Ensure all contract checks (icontract) and type hints are preserved or improved during refactor.
- If you add or change contracts, add/extend tests to cover new contract logic.
- Add protocol/ABC compliance and contract validation tests for all new abstractions.

### 6. No Placeholders
- Avoid leaving any TODOs or placeholders unless absolutely necessary, and document any that remain.

### 7. Final Validation
- After each major step, run all tests, mypy, and icontract demos.
- Run all major demos (e.g., `demo_hybrid_system.py`) to ensure end-to-end functionality.

---

## 📚 Abstraction Guidelines for Future Development

### Protocols and ABCs
- All new core abstractions must define and implement a protocol or ABC.
- Protocols should be placed in `app/core/protocols.py` and decorated with `@runtime_checkable`.
- Use explicit protocol/ABC inheritance in all concrete classes.
- Add runtime `assert isinstance(obj, ProtocolType)` checks where appropriate.

### Data Model Patterns
- **Core Logic:** Use `@dataclass` for all internal data structures.
- **API Boundaries:** Use Pydantic models for request/response validation.
- **Type Hints:** Use `TypedDict` only for type annotations.
- **Conversion:** Provide and test utilities for dataclass ↔ Pydantic conversion where needed.

### Testing and Validation
- All new abstractions must have protocol compliance and contract validation tests.
- Run the full test suite, mypy, and icontract validation after each change.
- Add/extend tests for all conversion utilities.

### Documentation and CI/CD
- Update docstrings and markdown docs to reflect new abstractions and model usage.
- Ensure CI/CD enforces type safety and protocol compliance.

### No Placeholders
- Avoid leaving TODOs or placeholders in production code. If unavoidable, document them clearly and track for resolution.

### Periodic Review
- Schedule regular abstraction consistency reviews to ensure ongoing compliance as the codebase evolves.

---

## 🚀 Immediate Script Usage Guidelines
- All demo scripts are now in the `demo/` directory (e.g., `demo/demo_abstractions.py`).
- Example and persistence scripts are in the `examples/` directory (e.g., `examples/persistence_examples_overview.py`).
- Update all usage instructions and code snippets to use the correct script paths.
- REST API endpoints do not use the `/api` prefix (e.g., `/concepts`, `/health`).

---

**For onboarding and future contributors:**
- Follow this workflow and these guidelines to maintain the architectural integrity, type safety, and maintainability of the ltnexp03 codebase.
- When in doubt, prefer explicit protocol/ABC usage and dataclass-based models for all new core logic.
