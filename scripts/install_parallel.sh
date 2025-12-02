#!/usr/bin/env bash
# ============================================================
#  Install GNU Parallel
#  Helper script to install GNU Parallel on various systems
# ============================================================

set -e

echo "=================================="
echo "GNU Parallel Installation Helper"
echo "=================================="
echo ""

# Detect OS
if [[ -f /etc/os-release ]]; then
    . /etc/os-release
    OS=$ID
else
    echo "Cannot detect OS. Please install manually."
    exit 1
fi

echo "Detected OS: $OS"
echo ""

case "$OS" in
    fedora|rhel|centos)
        echo "Installing GNU Parallel via DNF..."
        sudo dnf install -y parallel
        ;;
    ubuntu|debian)
        echo "Installing GNU Parallel via APT..."
        sudo apt-get update
        sudo apt-get install -y parallel
        ;;
    arch|manjaro)
        echo "Installing GNU Parallel via Pacman..."
        sudo pacman -S --noconfirm parallel
        ;;
    opensuse*)
        echo "Installing GNU Parallel via Zypper..."
        sudo zypper install -y parallel
        ;;
    *)
        echo "Unsupported OS: $OS"
        echo ""
        echo "Please install GNU Parallel manually:"
        echo "  - Homepage: https://www.gnu.org/software/parallel/"
        echo "  - Or via your package manager"
        exit 1
        ;;
esac

echo ""
echo "Verifying installation..."
if command -v parallel &> /dev/null; then
    PARALLEL_VERSION=$(parallel --version | head -1)
    echo "✓ GNU Parallel installed successfully!"
    echo "  Version: $PARALLEL_VERSION"
    echo ""
    echo "First-time setup: Accept citation notice"
    echo "Run: parallel --citation"
    echo "Then type 'will cite' and press Enter"
else
    echo "✗ Installation failed. Please install manually."
    exit 1
fi

echo ""
echo "=================================="
echo "Installation complete!"
echo "=================================="
