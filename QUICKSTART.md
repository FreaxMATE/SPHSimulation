# Quick Start Guide - SPH Simulation

This guide helps you get the SPH Simulation project running without Nix.

## 🚀 1-Minute Setup

### Linux/macOS:
```bash
git clone https://github.com/FreaxMATE/SPHSimulation.git
cd SPHSimulation
./setup.sh
```

### Windows:
```cmd
git clone https://github.com/FreaxMATE/SPHSimulation.git
cd SPHSimulation
setup.bat
```

## 📋 Prerequisites

- **Python 3.9+** (3.12 recommended)
- **Git** 
- **FFmpeg** (for video generation)

### Installing Prerequisites:

**Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install python3 python3-pip python3-venv ffmpeg git
```

**macOS (with Homebrew):**
```bash
brew install python ffmpeg git
```

**Windows:**
- Install Python from [python.org](https://python.org) (check "Add to PATH")
- Install Git from [git-scm.com](https://git-scm.com)
- Install FFmpeg: Download from [ffmpeg.org](https://ffmpeg.org) or use `choco install ffmpeg`

## 🔄 Alternative Setup Methods

### Method 1: pip + venv (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
# OR venv\Scripts\activate.bat  # Windows
pip install -r requirements.txt
```

### Method 2: Conda
```bash
conda env create -f environment.yml
conda activate sph-simulation
```

### Method 3: pip (Global - Not recommended)
```bash
pip install numpy pandas matplotlib scipy moviepy ffmpeg-python
```

## 🎮 Running Simulations

**Activate your environment first:**
```bash
source venv/bin/activate  # If using venv
# OR
conda activate sph-simulation  # If using conda
```

**Run simulations:**
```bash
# Sod shock tube simulation
cd 1/
python sodshock.py

# Planetary simulation  
cd 2/
python planets.py
```

## 🆘 Troubleshooting

### "Python not found"
- Ensure Python 3.9+ is installed and in your PATH
- Try `python3` instead of `python`

### "FFmpeg not found" 
- Install FFmpeg for your system
- On Windows, ensure FFmpeg is in your PATH

### "Module not found"
- Ensure you activated your virtual environment
- Re-run the setup script

### Permission denied (Linux/macOS)
```bash
chmod +x setup.sh
```

## 📦 What Gets Installed

- **NumPy**: Numerical computing
- **Matplotlib**: Plotting and visualization  
- **SciPy**: Scientific computing
- **Pandas**: Data manipulation
- **MoviePy**: Video processing
- **FFmpeg**: Video encoding (system dependency)

## 🎯 Expected Output

After running simulations, you should see:
- Real-time plots during simulation
- Generated MP4 animation files
- Console output showing simulation progress

---

For detailed information, see the main [README.md](README.md).