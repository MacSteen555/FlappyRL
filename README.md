# FlappyRL

A from-scratch C++20 reinforcement learning project: a Deep Q-Network (DQN) learns to play
a Flappy-Bird-style game. No ML libraries — the neural network, backprop, Adam optimizer,
replay buffer and training loop are all implemented here.

## Results

With the default settings, training takes 1–3 minutes on a desktop CPU. Across two seeds,
the greedy policy reached the evaluation cap (277 pipes, a 20,000-frame episode) on all 10
held-out pipe layouts by roughly 125k–225k environment steps.

DQN performance oscillates during training, so `app_train` evaluates periodically and keeps
the **best** checkpoint, not just the final one.

## Building

### Prerequisites
- CMake 3.15+
- A C++20 compiler (GCC, Clang, or MSVC; MinGW-w64 on Windows)
- SDL2 (optional; only needed for `app_play`)

### Build

```powershell
cmake -S . -B build -G "MinGW Makefiles" -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Binaries land in `build/bin/`. On Windows, `SDL2.dll` is copied next to `app_play.exe`
automatically.

### SDL2 setup (Windows / MinGW)

1. Download `SDL2-devel-2.x.x-mingw.tar.gz` from https://github.com/libsdl-org/SDL/releases
2. Extract to `C:\dev\SDL2` (picked up automatically), or anywhere and set `SDL2_ROOT`
3. Re-run CMake

Or with MSYS2: `pacman -S mingw-w64-x86_64-SDL2`. On Linux/macOS, SDL2 is found via
`find_package` / `pkg-config`.

## Usage

### Train an agent

```powershell
.\build\bin\app_train.exe
```

| Flag | Default | Meaning |
|---|---|---|
| `--steps N` | 300000 | Environment steps to train for |
| `--warmup N` | 5000 | Random-policy steps before learning starts |
| `--eval-every N` | 25000 | Steps between greedy evaluations |
| `--eval-episodes N` | 10 | Held-out layouts per evaluation |
| `--seed N` | 1 | RNG seed (network init, exploration, layouts) |
| `--out PATH` | `models/flappy_dqn.bin` | Best checkpoint; final weights go to `PATH.last` |

### Watch it play

```powershell
.\build\bin\app_play.exe --agent models\flappy_dqn.bin
```

### Play yourself

```powershell
.\build\bin\app_play.exe            # optional: --seed N
```

**Controls:** `SPACE` flap · `R` restart on a new layout · `ESC`/`Q` or close the window to quit

### Run the tests

```powershell
.\build\bin\FlappyRL_tests.exe      # or: ctest --test-dir build
```

## How it works

**Environment** (`env_flappy`): deterministic, seeded 2D physics at 60 Hz. The bird is a
point at fixed x; pipes scroll left. Rewards are +1 for passing a pipe's center and -1 for
hitting a pipe, the floor, or the ceiling.

**Observation** (4 floats): bird height, vertical velocity, horizontal distance to the
current pipe, and vertical offset to its gap center. The agent rescales these to ~[-1, 1].

**Agent** (`rl_dqn`):
- MLP `4 → 64 → 64 → 2` (ReLU), Q-value per action (no-flap, flap)
- Double DQN targets with a periodically synced target network
- Huber loss, Adam optimizer, batch gradients averaged over 64 samples
- Uniform experience replay (100k transitions)
- ε-greedy exploration decaying 1.0 → 0.01. Random actions flap only 10% of the time:
  at 60 Hz a 50/50 coin flip pins the bird to the ceiling, so it never reaches a pipe.
- Time-limit truncation is not treated as death, so long survival isn't punished

**Checkpoints** are a small binary format (magic `FRL1`, layer sizes, weights, biases) that
records its own architecture, so `app_play` can load any trained model.

## Project structure

```
FlappyRL/
├── include/ src/
│   ├── env_flappy/   # Game physics, collisions, rewards
│   ├── rl_dqn/       # Network, Adam, replay buffer, DQN agent
│   ├── render_sdl/   # SDL2 rendering + input
│   ├── app_train/    # Training CLI
│   └── app_play/     # Interactive play / watch a trained agent
└── tests/            # Catch2: gradient check, agent, checkpoints, environment
```
