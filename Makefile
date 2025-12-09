.PHONY: help build build-release test test-mock clippy clippy-fix fmt fmt-check clean package publish-dry publish install check-all

# Default target
help:
	@echo "trtx-rs Makefile"
	@echo "================"
	@echo ""
	@echo "Common targets:"
	@echo "  make build          - Build in debug mode with mock feature"
	@echo "  make build-release  - Build in release mode with mock feature"
	@echo "  make test           - Run all tests with mock feature"
	@echo "  make clippy         - Run clippy lints"
	@echo "  make fmt            - Format code"
	@echo "  make clean          - Clean build artifacts"
	@echo "  make check-all      - Run all checks (fmt, clippy, test)"
	@echo ""
	@echo "Publishing targets:"
	@echo "  make package        - Package both crates for publishing"
	@echo "  make publish-dry    - Dry-run publish to crates.io"
	@echo "  make publish        - Publish to crates.io (requires CRATES_IO_TOKEN)"
	@echo ""
	@echo "Development targets:"
	@echo "  make fmt-check      - Check code formatting"
	@echo "  make clippy-fix     - Auto-fix clippy warnings"

# Build targets
build:
	cargo build --features mock

build-release:
	cargo build --release --features mock

# Test targets
test:
	cargo test --features mock

test-mock:
	cargo test --features mock --verbose

# Linting targets
clippy:
	cargo clippy --features mock --all-targets -- -D warnings

clippy-fix:
	cargo clippy --fix --features mock --all-targets --allow-dirty

# Formatting targets
fmt:
	cargo fmt --all

fmt-check:
	cargo fmt --all -- --check

# Clean target
clean:
	cargo clean

# Run all checks (useful before committing)
check-all: fmt-check clippy test
	@echo "✓ All checks passed!"

# Packaging targets
package:
	@echo "Packaging trtx-sys..."
	cargo package --features mock -p trtx-sys --allow-dirty
	@echo "Packaging trtx..."
	cargo package --features mock -p trtx --allow-dirty
	@echo "✓ Both packages created successfully"

# Publishing targets
publish-dry:
	@echo "Dry-run publishing trtx-sys..."
	cargo publish --dry-run --features mock -p trtx-sys
	@echo "Dry-run publishing trtx..."
	cargo publish --dry-run --features mock -p trtx
	@echo "✓ Dry-run completed successfully"

publish:
	@echo "Publishing trtx-sys to crates.io..."
	cargo publish --features mock -p trtx-sys
	@echo "Waiting 30 seconds for trtx-sys to be available..."
	@sleep 30
	@echo "Publishing trtx to crates.io..."
	cargo publish --features mock -p trtx
	@echo "✓ Both crates published successfully!"

# Install from local (for testing)
install:
	cargo install --path trtx --features mock --force

# Run example
run-example:
	cargo run --features mock --example rustnn_executor
