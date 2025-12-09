#!/usr/bin/env pwsh
# Install git hooks for Windows

Write-Host "Installing git hooks..." -ForegroundColor Cyan

# Set git hooks directory
git config core.hooksPath .githooks

Write-Host "Git hooks installed successfully!" -ForegroundColor Green
Write-Host "Pre-commit hook will check code formatting before each commit." -ForegroundColor Green
