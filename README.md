# SPH Simulation Project

A collection of Smoothed Particle Hydrodynamics (SPH) simulations implemented in Python for computational physics applications.

## 🌟 Overview

This project implements SPH methods to simulate fluid dynamics problems, featuring:
- **Sod Shock Tube** simulations with animated visualizations
- **Planetary/Astrophysical** fluid dynamics scenarios
- Real-time visualization and animation generation
- Multiple implementation approaches (current and legacy methods)

## 📁 Project Structure

```
SPHSimulation/
├── 1/                          # Sod Shock Tube Simulations
│   ├── sodshock.py            # Main SPH shock tube implementation
│   ├── sodshock_leg.py        # Legacy implementation (v1)
│   ├── sodshock_leg2.py       # Legacy implementation (v2)
│   └── sodshock_movie.mp4     # Generated animation output
├── 2/                          # Planetary/Astrophysical Simulations
│   └── planets.py             # SPH simulation for planetary scenarios
├── flake.nix                  # Nix development environment
├── flake.lock                 # Nix lockfile
└── README.md                  # This file
```

## 🚀 Features

### Sod Shock Tube Simulation
- Simulates the classic Riemann problem in fluid dynamics
- Visualizes the evolution of:
  - **Density** (ρ)
  - **Pressure** (p) 
  - **Velocity** (v)
  - **Internal Energy** (e)
- Generates MP4 animations showing temporal evolution
- Includes legacy implementations for comparison and development history

### SPH Implementation
- Particle-based fluid simulation using smoothed particle hydrodynamics
- Adaptive time stepping with RK4 integration
- Kernel-based interpolation for smooth field calculations
- Equation of state: ideal gas law with γ = 1.4

## 🛠️ Dependencies

This project uses the following Python packages:
- `numpy` - Numerical computations
- `matplotlib` - Plotting and visualization
- `scipy` - Scientific computing and integration
- `moviepy` / `ffmpeg` - Animation generation

## 💻 Development Environment

The project includes a Nix flake for reproducible development environments:

```bash
# Enter development shell
nix develop
```

This provides all necessary Python packages and FFmpeg for animation generation.

## 🎯 Usage

### Running Sod Shock Simulations

```bash
# Navigate to directory 1
cd 1/

# Run main simulation
python sodshock.py

# Run legacy implementations
python sodshock_leg.py
python sodshock_leg2.py
```

### Running Planetary Simulations

```bash
# Navigate to directory 2
cd 2/

# Run planetary simulation
python planets.py
```

## 📊 Output

- **Real-time plots**: Display simulation progress
- **Animated visualizations**: MP4 files showing temporal evolution
- **Data analysis**: Particle state histories for post-processing

## 🧮 Physics Background

### Smoothed Particle Hydrodynamics (SPH)
SPH is a computational method for simulating fluid flows by representing the fluid as a collection of particles. Each particle carries physical properties (mass, density, pressure, velocity) and interacts with nearby particles through a smoothing kernel.

### Sod Shock Tube
A classic test problem in computational fluid dynamics involving:
- Initial discontinuity in density and pressure
- Evolution into shock waves, contact discontinuities, and rarefaction waves
- Exact analytical solutions available for validation

## 📈 Applications

This simulation framework is suitable for:
- **Educational purposes**: Understanding SPH methods and fluid dynamics
- **Research**: Computational physics and astrophysics applications  
- **Validation**: Testing new SPH implementations against known solutions
- **Visualization**: Creating animations for presentations and publications

## 📄 License

This project is licensed under the terms specified in the LICENSE file.

## 🤝 Contributing

For questions or collaboration opportunities, please refer to the repository owner.

---

*Part of Applied Computational Physics and Machine Learning coursework*