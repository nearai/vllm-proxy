#!/bin/bash
# Script to compile requirements.txt to requirements.lock.txt with all transitive dependencies locked
# This ensures reproducible builds by pinning all dependency versions
#
# The generated requirements.lock.txt includes:
#   - All direct dependencies from requirements.txt
#   - All transitive dependencies with exact versions
#   - SHA256 hashes for each package (for verification and security)
#
# Usage:
#   ./compile-requirements.sh

set -e

# Check if requirements.txt exists
if [ ! -f requirements.txt ]; then
    echo "Error: requirements.txt not found" >&2
    exit 1
fi

# Check if pip-tools is installed
if ! command -v pip-compile &> /dev/null; then
    echo "pip-tools not found. Installing pip-tools..."
    pip install pip-tools
fi

echo "Compiling requirements.txt to requirements.lock.txt..."
echo "This will resolve all transitive dependencies and may take a few minutes..."
echo ""

pip-compile \
    --generate-hashes \
    --output-file requirements.lock.txt \
    --resolver=backtracking \
    --allow-unsafe \
    requirements.txt

echo ""
echo "✓ Successfully generated requirements.lock.txt with all dependencies locked"
echo ""
echo "Summary:"
echo "  - Source: requirements.txt (direct dependencies)"
echo "  - Output: requirements.lock.txt (all transitive dependencies with hashes)"
echo ""
echo "Next steps:"
echo "  1. Review the changes in requirements.lock.txt"
echo "  2. Commit both requirements.txt and requirements.lock.txt to version control"
echo ""
echo "To update dependencies in the future:"
echo "  1. Edit requirements.txt with new versions or packages"
echo "  2. Run: ./compile-requirements.sh"
echo "  3. Review and commit the updated files"
