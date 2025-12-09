# Git Hooks

This directory contains git hooks that help maintain code quality.

## Pre-commit Hook

The pre-commit hook checks code formatting using `cargo fmt` before allowing a commit.

### Installation

**Unix/Linux/macOS:**
```bash
./.githooks/install.sh
```

**Windows:**
```powershell
powershell -ExecutionPolicy Bypass -File .githooks/install.ps1
```

**Manual installation:**
```bash
git config core.hooksPath .githooks
```

### What it does

Before each commit, the hook will:
1. Run `cargo fmt --all -- --check`
2. If formatting issues are found, the commit is blocked
3. You'll need to run `cargo fmt --all` to fix the issues and try again

### Bypassing the hook

If you need to bypass the pre-commit hook (not recommended):
```bash
git commit --no-verify
```
