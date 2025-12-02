#!/usr/bin/env bash
# ============================================================
#  Install GNU Parallel Locally (no sudo required)
#  Downloads and installs parallel in the project directory
# ============================================================

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
INSTALL_DIR="${PROJECT_ROOT}/.local"
BIN_DIR="${INSTALL_DIR}/bin"
PARALLEL_VERSION="20231122"
PARALLEL_URL="https://ftp.gnu.org/gnu/parallel/parallel-${PARALLEL_VERSION}.tar.bz2"

echo "=================================="
echo "GNU Parallel Local Installation"
echo "=================================="
echo ""
echo "This will install GNU Parallel locally in:"
echo "  ${INSTALL_DIR}"
echo ""

# Create directories
mkdir -p "${INSTALL_DIR}/src"
mkdir -p "${BIN_DIR}"

# Download
echo "Downloading GNU Parallel ${PARALLEL_VERSION}..."
cd "${INSTALL_DIR}/src"
if [ ! -f "parallel-${PARALLEL_VERSION}.tar.bz2" ]; then
    curl -L -o "parallel-${PARALLEL_VERSION}.tar.bz2" "${PARALLEL_URL}"
    echo "✓ Downloaded"
else
    echo "✓ Already downloaded"
fi

# Extract
echo "Extracting..."
if [ ! -d "parallel-${PARALLEL_VERSION}" ]; then
    tar xjf "parallel-${PARALLEL_VERSION}.tar.bz2"
    echo "✓ Extracted"
else
    echo "✓ Already extracted"
fi

# Install
echo "Installing to ${INSTALL_DIR}..."
cd "parallel-${PARALLEL_VERSION}"
./configure --prefix="${INSTALL_DIR}" > /dev/null 2>&1
make -j$(nproc) > /dev/null 2>&1
make install > /dev/null 2>&1

echo "✓ Installed"
echo ""

# Verify
if [ -f "${BIN_DIR}/parallel" ]; then
    VERSION=$("${BIN_DIR}/parallel" --version | head -1)
    echo "=================================="
    echo "Installation Successful!"
    echo "=================================="
    echo ""
    echo "Installed: ${VERSION}"
    echo "Location: ${BIN_DIR}/parallel"
    echo ""
    echo "The scripts will automatically use this local installation."
    echo ""
    echo "First-time setup: Accept citation notice"
    echo "Run: ${BIN_DIR}/parallel --citation"
    echo "Then type 'will cite' and press Enter"
    echo ""
else
    echo "✗ Installation failed"
    exit 1
fi
