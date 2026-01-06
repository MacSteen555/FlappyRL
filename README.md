# FlappyRL

A modern C++ reinforcement learning engine where a DQN agent learns to play a Flappy-Bird-style game.

## Project Status

✅ **Completed:**
- CMake project setup (C++20, Release/Debug configs)
- Flappy Bird environment (deterministic physics, collision detection, rewards)
- SDL2 visualization with keyboard controls
- Interactive gameplay (`app_play`)

🚧 **In Progress:**
- Neural network implementation
- Replay buffer
- Adam optimizer
- DQN agent
- Training loop

## Features

- **Custom 2D Physics Environment** - Deterministic Flappy Bird simulation
- **SDL2 Visualization** - Real-time rendering with interactive controls
- **DQN Implementation** (planned) - Deep Q-Network for learning
- **Replay Buffer** (planned) - Experience storage for training
- **Adam Optimizer** (planned) - Adaptive gradient descent
- **Multi-threaded Collection** (future) - Parallel experience gathering
- **CUDA Integration** (future) - GPU acceleration

## Building

### Prerequisites
- CMake 3.15+
- C++20 compiler (GCC/Clang/MSVC)
- SDL2 (optional, for visualization)
- MinGW (Windows) or standard compiler

### Build Instructions

```powershell
# Configure
cd build
cmake .. -G "MinGW Makefiles"

# Build
cmake --build .

# Run visualization
.\bin\app_play.exe
```

### SDL2 Setup (Windows/MinGW)

1. Download SDL2 from https://github.com/libsdl-org/SDL/releases
2. Extract to `C:\dev\SDL2`
3. Set environment variable: `$env:SDL2_ROOT = "C:\dev\SDL2"`
4. Reconfigure CMake

## Usage

### Play the Game
```powershell
.\bin\app_play.exe
```

**Controls:**
- `SPACE` - Flap (tap, don't hold)
- `R` - Restart after game over
- `ESC/Q` - Quit

### Train the Agent (when implemented)
```powershell
.\bin\app_train.exe
```

## Project Structure

```
FlappyRL/
├── include/          # Header files
│   ├── env_flappy/  # Environment interface
│   ├── rl_dqn/      # DQN components (in progress)
│   └── render_sdl/  # SDL2 rendering
├── src/             # Implementation files
├── tests/           # Unit tests
└── scripts/         # Utility scripts
```

## Next Steps

See [PROJECT_STATUS.md](PROJECT_STATUS.md) for detailed status and roadmap.
