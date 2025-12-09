#!/bin/sh
# Install git hooks

echo "Installing git hooks..."

# Set git hooks directory
git config core.hooksPath .githooks

echo "Git hooks installed successfully!"
echo "Pre-commit hook will check code formatting before each commit."
