# High-Performance Evolutionary Simulator Engine: CUDA Evolution

A very high-performance simulator engine written in Python that uses custom CUDA kernels, OpenGL, and PyTorch to minimize device-host data transfer. The simulator provides a general framework for users to define custom environments and entities. These entities inhabit the environments and evolve over time via natural selection.

![Visualization of a CUDA-Evolution simulation](/assets/cuda-evolution.png)

## Features

- **Custom CUDA Kernels**: Optimized CUDA kernels for high-performance computation.
- **OpenGL Integration**: Real-time rendering of environments and entities.
- **PyTorch Support**: Seamless integration with PyTorch for machine learning and reinforcement learning workflows.
- **Device-Host Optimization**: Minimal data transfer between CPU and GPU, ensuring fast execution.
- **Customizable Environments**: Users can define their own environments and the rules that govern them.
- **Evolving Entities**: Entities evolve through natural selection based on the laws of the environment.

# Design

The basis of a CUDA-Evolution world is the `evolution.core.gworld.GWorld` object. The `GWorld.step` function defines how the world changes with each time step. The `GWorld` manages two primary things:
- The "food grid", a 2-dimensional tensor where each value represents the current food level of a location in the environment.
- The `Creatures` object, defined in `evolution/core/creatures/creature.py`. `Creature` objects consist of a series of `CreatureTraits` that define attributes of the creatures in the simulation. The initialization method of different `CreatureTraits` can be specified, whether they should be set to a pre-defined value, sampled from some distribution, inherited from their parents, or be a function of some other `CreatureTrait`.

Many factors about the simulation, such as food growth rates or costs for doing various actions are managed using a centralized config object, located at `evolution.core.config.Config`.

## Mechanics

### Brains / Interactivity
Creatures' only way of interacting with the world are through movement and rotation. How much a creature rotates and/or moves on a given step is decided by the output of that creature's feed-forward neural network brain. These brains have additional outputs that correspond to "memory" units. These outputs are simply fed back in to the creature's brain at the next time step as inputs, to theoretically allow some capacity for memory.


### Brain Inputs

Creatures have a set number of vision rays that extend from the center of their body. Rays that intersect other creatures result in an input being added to that creature's brain corresponding to the color of the creature the ray intersected. Creatures also can see the food level of the food grid cells in a small square around themselves. Additional inputs to the neural network brain include the health, energy, head direction, size, and the memory.

### Reproduction

As creatures roam around the food grid, collecting food, their internal energy increases. This energy is necessary for staying alive, as well as taking actions. If this energy increases above a threshold that is increasing in the creature's size, then that creature is allowed to reproduce. The reproduction process involves copying its "mutable" traits, applying random perturbations to them, and then creating a new creature in roughly the same area with those perturbed traits.

### Attacking

Creatures have a rectangular "attacking hitbox" outside of their central, circular body with which attacks may be made. If their attacking hitbox intersects another creature's body, then an attack is made. The damage of an attack scales with creature size, and creatures that are similar in color do significantly reduced damage to each other. Attacking will reduce the health of the targeted creature and will cost the attacking creature some energy.

### Death

If a creature's health or energy drops below 0, that creature is dead. Dead creatures are removed from the simulation. When a creature dies, it increases the food level of the food grid by an amount that scales with the size of the creature.

### Ageing

Creatures go through three distinct stages of life. In adolescence, the first stage, creatures cannot reproduce, but can do pretty much everything else. In maturity, creatures can reproduce. In old age, creatures can still reproduce, but have an increasingly hard time collecting energy, and they take more damage for the same attacks.

# Design Goals

1. Evolution of enviroment must lead to interesting behaviour developing, with multiple, distinct survival strategies possible.
2. Simulation must be highly performant, able to support hundreds of thousands to millions of complex creatures in massive enviroments.
3. Code must be well-documented, easy to understand, and use good design principles.


# Getting Started

## Installation

Via pip:

```bash
git clone https://github.com/JacksonKaunismaa/evolution.git
cd evolution
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Via conda:
```bash
git clone https://github.com/JacksonKaunismaa/evolution.git
cd evolution
conda create -n <name> --file=environment.yml
```


## Usage
Launch a simulation:

```bash
python3 ./launch.py
```

Simulation parameters can be modified by opening launch.py, and changing the `evolution.core.config.Config` object. Documentation for all supported parameters are outlined in `evolution/core/config.py`.

To do nsys-style benchmarking, use the following command:

```bash
nsys profile -o <log_name> -w true -t cuda,nvtx,osrt,cudnn,cublas -s none  --capture-range=cudaProfilerApi --capture-range-end=stop -x true ./launch.py --style nsys
```

Next, open `log_name.nsys-rep` in NVIDIA NSight Compute and look at the CUDA HW row.


# Planned Features
 - RL to allow creatures to learn much quicker.
 - Non-uniform growth of food grid to incentivize more sophisticated navigation.
 - Large-scale, slow moving environmental changes
 - Increase the number of plots, with potential to define custom plots / tracking objectives on the fly.
 - Plot manager to enable better GUI resizing, especially to adjust for different monitor resolutions.
 - Sexual reproduction
 - Unique food sources based on creature species to allow different niches to co-exist.