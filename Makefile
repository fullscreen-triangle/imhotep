# Imhotep Neural Network Framework Makefile
# High-Performance Specialized Neural Network Framework

.PHONY: help build clean test bench docs install dev-install check format lint
.PHONY: python-build python-install python-test wasm-build cuda-build
.PHONY: docker-build docker-run release

# Default target
.DEFAULT_GOAL := help

# Variables
RUST_VERSION := 1.70
PYTHON_VERSION := 3.9
CARGO := cargo
PYTHON := python3
PIP := pip3
MATURIN := maturin

# Build configurations
BUILD_TYPE ?= release
FEATURES ?= default
TARGET_DIR := target
PYTHON_DIR := python
WASM_DIR := pkg

# Platform detection
UNAME_S := $(shell uname -s)
ifeq ($(UNAME_S),Linux)
    PLATFORM := linux
    LIB_EXT := so
endif
ifeq ($(UNAME_S),Darwin)
    PLATFORM := macos
    LIB_EXT := dylib
endif
ifeq ($(OS),Windows_NT)
    PLATFORM := windows
    LIB_EXT := dll
endif

# Color codes for pretty output
GREEN := \033[0;32m
BLUE := \033[0;34m
YELLOW := \033[1;33m
RED := \033[0;31m
NC := \033[0m # No Color

help: ## Show this help message
	@echo "$(GREEN)Imhotep Neural Network Framework$(NC)"
	@echo "$(BLUE)High-Performance Specialized Neural Network Framework$(NC)"
	@echo ""
	@echo "$(YELLOW)Available targets:$(NC)"
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  $(GREEN)%-18s$(NC) %s\n", $$1, $$2}' $(MAKEFILE_LIST)

# Development setup
dev-setup: ## Set up development environment
	@echo "$(BLUE)Setting up development environment...$(NC)"
	rustup install $(RUST_VERSION)
	rustup default $(RUST_VERSION)
	rustup component add clippy rustfmt
	$(PIP) install maturin[patchelf] pytest black isort mypy
	$(PIP) install -e .[dev]

install-tools: ## Install additional development tools
	@echo "$(BLUE)Installing development tools...$(NC)"
	$(CARGO) install cargo-watch cargo-audit cargo-outdated
	$(CARGO) install flamegraph
	$(PIP) install pre-commit
	pre-commit install

# Build targets
build: ## Build the Rust library (release mode)
	@echo "$(BLUE)Building Rust library...$(NC)"
	$(CARGO) build --release --features $(FEATURES)

build-dev: ## Build the Rust library (debug mode)
	@echo "$(BLUE)Building Rust library (debug)...$(NC)"
	$(CARGO) build --features $(FEATURES)

build-all: ## Build all variants (release, debug, with all features)
	@echo "$(BLUE)Building all variants...$(NC)"
	$(CARGO) build --release
	$(CARGO) build --release --features max-performance
	$(CARGO) build --release --features safe

# Python bindings
python-build: ## Build Python bindings
	@echo "$(BLUE)Building Python bindings...$(NC)"
	$(MATURIN) build --release --features python

python-develop: ## Install Python bindings in development mode
	@echo "$(BLUE)Installing Python bindings (development)...$(NC)"
	$(MATURIN) develop --features python

python-install: python-build ## Install Python bindings
	@echo "$(BLUE)Installing Python bindings...$(NC)"
	$(PIP) install target/wheels/*.whl --force-reinstall

python-test: ## Run Python tests
	@echo "$(BLUE)Running Python tests...$(NC)"
	$(PYTHON) -m pytest $(PYTHON_DIR)/tests/ -v

# WebAssembly build
wasm-build: ## Build WebAssembly module
	@echo "$(BLUE)Building WebAssembly module...$(NC)"
	wasm-pack build --target web --features wasm
	wasm-pack build --target nodejs --features wasm --out-dir pkg-node

# CUDA build
cuda-build: ## Build with CUDA support
	@echo "$(BLUE)Building with CUDA support...$(NC)"
	$(CARGO) build --release --features cuda

# Testing
test: ## Run all Rust tests
	@echo "$(BLUE)Running Rust tests...$(NC)"
	$(CARGO) test --features $(FEATURES)

test-all: ## Run tests with all feature combinations
	@echo "$(BLUE)Running tests with all features...$(NC)"
	$(CARGO) test --all-features
	$(CARGO) test --no-default-features

test-integration: ## Run integration tests
	@echo "$(BLUE)Running integration tests...$(NC)"
	$(CARGO) test --test integration --features $(FEATURES)

test-doc: ## Test documentation examples
	@echo "$(BLUE)Testing documentation examples...$(NC)"
	$(CARGO) test --doc

# Benchmarking
bench: ## Run benchmarks
	@echo "$(BLUE)Running benchmarks...$(NC)"
	$(CARGO) bench --features benchmarks

bench-compare: ## Run benchmarks and compare with baseline
	@echo "$(BLUE)Running comparative benchmarks...$(NC)"
	$(CARGO) bench --features benchmarks -- --save-baseline main

profile: ## Profile the application
	@echo "$(BLUE)Profiling application...$(NC)"
	$(CARGO) build --release --features max-performance
	valgrind --tool=callgrind --callgrind-out-file=callgrind.out \
		./target/release/imhotep-cli --profile
	kcachegrind callgrind.out

# Code quality
check: ## Run cargo check
	@echo "$(BLUE)Running cargo check...$(NC)"
	$(CARGO) check --all-targets --all-features

clippy: ## Run clippy linter
	@echo "$(BLUE)Running clippy...$(NC)"
	$(CARGO) clippy --all-targets --all-features -- -D warnings

format: ## Format code
	@echo "$(BLUE)Formatting Rust code...$(NC)"
	$(CARGO) fmt --all
	@echo "$(BLUE)Formatting Python code...$(NC)"
	black $(PYTHON_DIR)/
	isort $(PYTHON_DIR)/

format-check: ## Check code formatting
	@echo "$(BLUE)Checking Rust formatting...$(NC)"
	$(CARGO) fmt --all -- --check
	@echo "$(BLUE)Checking Python formatting...$(NC)"
	black --check $(PYTHON_DIR)/
	isort --check $(PYTHON_DIR)/

lint: clippy ## Run all linters
	@echo "$(BLUE)Running mypy...$(NC)"
	mypy $(PYTHON_DIR)/

audit: ## Security audit
	@echo "$(BLUE)Running security audit...$(NC)"
	$(CARGO) audit

outdated: ## Check for outdated dependencies
	@echo "$(BLUE)Checking for outdated dependencies...$(NC)"
	$(CARGO) outdated

# Documentation
docs: ## Build documentation
	@echo "$(BLUE)Building documentation...$(NC)"
	$(CARGO) doc --all-features --no-deps
	@echo "$(GREEN)Documentation built: target/doc/imhotep/index.html$(NC)"

docs-open: docs ## Build and open documentation
	@echo "$(BLUE)Opening documentation...$(NC)"
	$(CARGO) doc --all-features --no-deps --open

docs-python: ## Build Python documentation
	@echo "$(BLUE)Building Python documentation...$(NC)"
	cd $(PYTHON_DIR) && sphinx-build -b html docs docs/_build/html

# Release management
tag-release: ## Create a git tag for release (requires VERSION env var)
ifndef VERSION
	$(error VERSION is not set. Use: make tag-release VERSION=x.y.z)
endif
	@echo "$(BLUE)Creating release tag v$(VERSION)...$(NC)"
	git tag -a v$(VERSION) -m "Release version $(VERSION)"
	git push origin v$(VERSION)

release-rust: ## Build release artifacts for Rust
	@echo "$(BLUE)Building Rust release artifacts...$(NC)"
	$(CARGO) build --release --all-features
	$(CARGO) build --release --target x86_64-unknown-linux-gnu
	$(CARGO) build --release --target x86_64-apple-darwin
	$(CARGO) build --release --target x86_64-pc-windows-gnu

release-python: ## Build Python release artifacts
	@echo "$(BLUE)Building Python release artifacts...$(NC)"
	$(MATURIN) build --release --features python
	$(MATURIN) build --release --features python --target x86_64-unknown-linux-gnu
	$(MATURIN) build --release --features python --target x86_64-apple-darwin
	$(MATURIN) build --release --features python --target x86_64-pc-windows-gnu

release: release-rust release-python ## Build all release artifacts

# Docker
docker-build: ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t imhotep:latest .

docker-build-dev: ## Build development Docker image
	@echo "$(BLUE)Building development Docker image...$(NC)"
	docker build -f Dockerfile.dev -t imhotep:dev .

docker-run: ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	docker run -it --rm imhotep:latest

docker-shell: ## Open shell in Docker container
	@echo "$(BLUE)Opening shell in Docker container...$(NC)"
	docker run -it --rm imhotep:dev /bin/bash

# Continuous Integration
ci-test: ## Run CI test suite
	@echo "$(BLUE)Running CI test suite...$(NC)"
	$(MAKE) format-check
	$(MAKE) clippy  
	$(MAKE) test-all
	$(MAKE) python-test
	$(MAKE) docs

ci-bench: ## Run CI benchmarks
	@echo "$(BLUE)Running CI benchmarks...$(NC)"
	$(MAKE) bench

# Utility targets
clean: ## Clean build artifacts
	@echo "$(BLUE)Cleaning build artifacts...$(NC)"
	$(CARGO) clean
	rm -rf $(PYTHON_DIR)/build/
	rm -rf $(PYTHON_DIR)/dist/
	rm -rf $(PYTHON_DIR)/*.egg-info/
	rm -rf $(WASM_DIR)/
	rm -rf pkg-node/
	rm -f callgrind.out
	find . -name "*.pyc" -delete
	find . -name "__pycache__" -type d -exec rm -rf {} +

clean-all: clean ## Clean everything including caches
	@echo "$(BLUE)Deep cleaning...$(NC)"
	$(CARGO) clean
	rm -rf ~/.cargo/registry/cache/
	rm -rf target/
	$(PIP) cache purge

# Development workflow  
dev: ## Start development workflow (watch and test)
	@echo "$(BLUE)Starting development workflow...$(NC)"
	$(CARGO) watch -x "check --all-features" -x "test --all-features"

quick-test: ## Run quick tests (no integration tests)
	@echo "$(BLUE)Running quick tests...$(NC)"
	$(CARGO) test --lib --bins --features $(FEATURES)

install: build ## Install the binary locally
	@echo "$(BLUE)Installing imhotep binary...$(NC)"
	$(CARGO) install --path . --features $(FEATURES)

uninstall: ## Uninstall the binary
	@echo "$(BLUE)Uninstalling imhotep binary...$(NC)"
	$(CARGO) uninstall imhotep

# Performance testing  
perf-test: ## Run performance tests
	@echo "$(BLUE)Running performance tests...$(NC)"
	$(CARGO) test --release --features max-performance -- --ignored

memory-test: ## Run memory usage tests
	@echo "$(BLUE)Running memory tests...$(NC)"
	$(CARGO) build --release --features max-performance
	valgrind --tool=massif --massif-out-file=massif.out \
		./target/release/imhotep-cli --benchmark
	ms_print massif.out > memory-profile.txt
	@echo "$(GREEN)Memory profile saved to memory-profile.txt$(NC)"

# Examples
run-examples: ## Run all examples
	@echo "$(BLUE)Running examples...$(NC)"
	$(CARGO) run --example basic_neural_network --features $(FEATURES)
	$(CARGO) run --example oscillatory_dynamics --features $(FEATURES)
	$(CARGO) run --example quantum_processing --features $(FEATURES)

# Status information
status: ## Show project status
	@echo "$(GREEN)Imhotep Project Status$(NC)"
	@echo "$(YELLOW)Rust Version:$(NC) $(shell rustc --version)"
	@echo "$(YELLOW)Cargo Version:$(NC) $(shell cargo --version)"
	@echo "$(YELLOW)Python Version:$(NC) $(shell python3 --version)"
	@echo "$(YELLOW)Platform:$(NC) $(PLATFORM)"
	@echo "$(YELLOW)Features:$(NC) $(FEATURES)"
	@echo "$(YELLOW)Build Type:$(NC) $(BUILD_TYPE)"
	$(CARGO) tree --depth 1 