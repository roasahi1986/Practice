#!/usr/bin/env bash
# =============================================================================
# Build wheel for ml_platform package
# =============================================================================
# Usage: ./scripts/build_wheel.sh
#
# Output: dist/ml_platform-<version>-py3-none-any.whl
# =============================================================================

set -e  # Exit on error

# Get script directory (works in both bash and zsh)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_ROOT"

echo "=============================================="
echo "Building ml_platform wheel"
echo "=============================================="

# Clean previous builds
echo "Cleaning previous builds..."
rm -rf build/ dist/ ml_platform.egg-info
# Use find to remove any .egg-info directories (avoids zsh glob errors)
find . -maxdepth 1 -name "*.egg-info" -type d -exec rm -rf {} + 2>/dev/null || true

# Install build dependencies if needed
echo "Installing build dependencies..."
python3 -m pip install --quiet build

# Build the wheel
echo "Building wheel..."
python3 -m build --wheel

# Show the result
echo ""
echo "=============================================="
echo "Build complete!"
echo "=============================================="
echo ""
echo "Wheel location:"
ls -la dist/*.whl

echo ""
echo "To install locally:"
echo "  pip install dist/ml_platform-*.whl"
echo ""
echo "To install with local dev dependencies (includes pyspark):"
echo "  pip install 'dist/ml_platform-*.whl[local]'"
echo ""
echo "To upload to Databricks:"
echo "  1. Upload the .whl file to DBFS or workspace"
echo "  2. Install on cluster: %pip install /path/to/ml_platform-*.whl"

