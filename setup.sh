#!/bin/bash

# Setup script for SPH Simulation project
# Works on Ubuntu/Debian, macOS (with Homebrew), and other Linux distributions

set -e

echo "🚀 Setting up SPH Simulation project..."

# Function to detect OS
detect_os() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        if command -v apt-get &> /dev/null; then
            echo "ubuntu"
        elif command -v yum &> /dev/null; then
            echo "rhel"
        elif command -v pacman &> /dev/null; then
            echo "arch"
        else
            echo "linux"
        fi
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        echo "macos"
    else
        echo "unknown"
    fi
}

# Function to install system dependencies
install_system_deps() {
    local os=$(detect_os)
    echo "📦 Installing system dependencies for $os..."
    
    case $os in
        ubuntu)
            sudo apt-get update
            sudo apt-get install -y python3 python3-pip python3-venv ffmpeg
            ;;
        rhel)
            sudo yum install -y python3 python3-pip ffmpeg
            ;;
        arch)
            sudo pacman -S python python-pip ffmpeg
            ;;
        macos)
            if command -v brew &> /dev/null; then
                brew install python ffmpeg
            else
                echo "❌ Homebrew not found. Please install Homebrew first:"
                echo "   /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
                exit 1
            fi
            ;;
        *)
            echo "⚠️  Unknown OS. Please install Python 3.9+ and FFmpeg manually."
            ;;
    esac
}

# Function to setup Python environment
setup_python_env() {
    echo "🐍 Setting up Python virtual environment..."
    
    # Create virtual environment
    python3 -m venv venv
    
    # Activate virtual environment
    source venv/bin/activate
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install Python dependencies
    echo "📚 Installing Python packages..."
    pip install -r requirements.txt
    
    echo "✅ Python environment setup complete!"
}

# Function to verify installation
verify_installation() {
    echo "🔍 Verifying installation..."
    
    source venv/bin/activate
    
    python3 -c "
import numpy as np
import matplotlib.pyplot as plt
import scipy
import moviepy
print('✅ All Python packages imported successfully!')
print(f'NumPy version: {np.__version__}')
print(f'Matplotlib version: {plt.matplotlib.__version__}')
print(f'SciPy version: {scipy.__version__}')
print(f'MoviePy version: {moviepy.__version__}')
"
    
    # Check ffmpeg
    if command -v ffmpeg &> /dev/null; then
        echo "✅ FFmpeg is available: $(ffmpeg -version | head -n 1)"
    else
        echo "⚠️  FFmpeg not found in PATH. Video generation may not work."
    fi
}

# Main execution
main() {
    # Check if requirements.txt exists
    if [[ ! -f "requirements.txt" ]]; then
        echo "❌ requirements.txt not found. Please run this script from the project root directory."
        exit 1
    fi
    
    # Install system dependencies
    read -p "Install system dependencies? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        install_system_deps
    fi
    
    # Setup Python environment
    setup_python_env
    
    # Verify installation
    verify_installation
    
    echo ""
    echo "🎉 Setup complete!"
    echo ""
    echo "To activate the environment in the future, run:"
    echo "   source venv/bin/activate"
    echo ""
    echo "To run the simulations:"
    echo "   cd 1/ && python sodshock.py    # Sod shock tube simulation"
    echo "   cd 2/ && python planets.py     # Planetary simulation"
}

main "$@"