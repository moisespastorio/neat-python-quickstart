# 🧬 NEAT-Python — Essential Cheat Sheet & Tutorial

> For those who already know Python and have some ML background, but want to understand **neuroevolution** without the headache.

---

## 🤔 What is NEAT?

**NEAT** = *NeuroEvolution of Augmenting Topologies*

It's an algorithm that **evolves neural networks** using principles from biological evolution:
- **Natural selection** → the best survive
- **Mutation** → new connections and neurons appear randomly
- **Crossover** → two networks "reproduce" and generate offspring

> 💡 **Fun fact:** NEAT was created by Kenneth Stanley and Risto Miikkulainen in 2002. Its key innovation is that *the network topology also evolves* — you don't need to define how many layers or neurons to use. The network designs itself!

```
                    KEY DIFFERENCE
    
    Traditional ML:   You define the architecture → train weights
    NEAT:             Algorithm defines architecture + weights via evolution
```

---

## 📦 Installation

```bash
pip install neat-python
```

---

## 🗂️ Project Structure

```
my_project/
│
├── config-feedforward.txt   ← Configuration file (the "DNA" of your experiment)
├── main.py                  ← Your main code
└── checkpoint/              ← (optional) Automatic saves
```

---

## ⚙️ The Configuration File

This is the **heart of NEAT**. It controls everything: population size, mutation rates, fitness criteria, and more.

```ini
# config-feedforward.txt

[NEAT]
fitness_criterion     = max        # How to pick the best: max, min, or mean
fitness_threshold     = 300        # Target fitness to stop evolution
pop_size              = 150        # Population size (number of "brains")
reset_on_extinction   = False      # Restart if all species go extinct?

[DefaultGenome]
# Neurons
num_inputs              = 2        # How many inputs your network receives
num_outputs             = 1        # How many outputs it produces
num_hidden              = 0        # Initial hidden neurons (can be 0!)
feed_forward            = True     # True = feedforward / False = recurrent

# Activation functions
activation_default      = sigmoid
activation_mutate_rate  = 0.0
activation_options      = sigmoid

# Connections
conn_add_prob           = 0.5      # Prob. of adding a new connection
conn_delete_prob        = 0.5      # Prob. of removing a connection
node_add_prob           = 0.2      # Prob. of adding a new neuron
node_delete_prob        = 0.2      # Prob. of removing a neuron

# Weights
weight_init_mean        = 0.0
weight_init_stdev       = 1.0
weight_max_value        = 30
weight_min_value        = -30
weight_mutate_rate      = 0.8
weight_replace_rate     = 0.1

# Bias
bias_init_mean          = 0.0
bias_init_stdev         = 1.0
bias_max_value          = 30
bias_min_value          = -30
bias_mutate_rate        = 0.7
bias_replace_rate       = 0.1

[DefaultSpeciesSet]
compatibility_threshold = 3.0      # Genetic distance to be considered the "same species"

[DefaultStagnation]
species_fitness_func = max
max_stagnation       = 20          # Generations without improvement before killing the species
species_elitism      = 2           # Minimum number of species always preserved

[DefaultReproduction]
elitism            = 2             # Top N genomes always pass without mutation
survival_threshold = 0.2           # Top 20% of each species survives to reproduce
```

---

## 🧠 Essential Concepts (in 1 minute)

| Concept | What it is | Analogy |
|---------|-----------|---------|
| **Genome** | A neural network + its genes | One individual in the population |
| **Population** | Set of all genomes | The entire group |
| **Species** | Group of similar genomes | Friend groups with similar traits |
| **Fitness** | Performance score | Grade on a test |
| **Generation** | One round of evolution | One school semester |
| **Node Gene** | An encoded neuron | A neuron in the genome |
| **Connection Gene** | An encoded synapse | A connection in the genome |

---

## 🚀 Minimum Working Code

### Example problem: learning XOR

```python
import neat
import os

# ─────────────────────────────────────────
# 1. DEFINE THE FITNESS FUNCTION
#    Here you tell NEAT how good each genome is
# ─────────────────────────────────────────

# XOR data
xor_inputs  = [(0, 0), (0, 1), (1, 0), (1, 1)]
xor_outputs = [   0,      1,      1,      0  ]

def eval_genomes(genomes, config):
    """
    Receives ALL genomes of the current generation.
    You must calculate and assign the fitness of each one.
    """
    for genome_id, genome in genomes:

        # Create the neural network from the genome
        net = neat.nn.FeedForwardNetwork.create(genome, config)

        genome.fitness = 4.0  # maximum possible fitness

        for xi, xo in zip(xor_inputs, xor_outputs):
            output = net.activate(xi)              # Pass inputs through the network
            genome.fitness -= (output[0] - xo) ** 2  # Penalize errors


# ─────────────────────────────────────────
# 2. LOAD THE CONFIGURATION
# ─────────────────────────────────────────

def run(config_file):
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_file
    )

    # ─────────────────────────────────────────
    # 3. CREATE THE POPULATION
    # ─────────────────────────────────────────
    p = neat.Population(config)

    # Add terminal reporters (optional but useful)
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)

    # ─────────────────────────────────────────
    # 4. RUN THE EVOLUTION
    #    n = maximum number of generations
    # ─────────────────────────────────────────
    winner = p.run(eval_genomes, n=50)

    # ─────────────────────────────────────────
    # 5. USE THE WINNER
    # ─────────────────────────────────────────
    print(f'\nBest genome:\n{winner}')

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

    print("\n--- Testing the winner ---")
    for xi, xo in zip(xor_inputs, xor_outputs):
        output = winner_net.activate(xi)
        print(f"Input: {xi} | Expected: {xo} | Output: {output[0]:.3f}")


# ─────────────────────────────────────────
# ENTRY POINT
# ─────────────────────────────────────────
if __name__ == '__main__':
    local_dir   = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward.txt')
    run(config_path)
```

---

## 🔄 Execution Flow (the NEAT "loop")

```
                    ┌─────────────────────────────┐
                    │   Create initial population  │
                    │   (random genomes)           │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   For each genome:           │
                    │   • Create neural network    │  ← eval_genomes()
                    │   • Run in the environment   │
                    │   • Calculate fitness        │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   fitness >= threshold?      │
                    │         YES → STOP 🏆        │
                    │         NO  → continue ↓    │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Speciation                 │
                    │   (group similar genomes)    │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   Selection + Reproduction   │
                    │   (crossover + mutation)     │
                    └──────────────┬──────────────┘
                                   │
                    ┌──────────────▼──────────────┐
                    │   New generation             │
                    └──────────────┬──────────────┘
                                   │
                                   └──► (back to the top)
```

---

## 🕹️ Available Network Types

```python
# Feedforward (most common, simple, no loops)
net = neat.nn.FeedForwardNetwork.create(genome, config)

# Recurrent (has memory, good for sequences/time)
net = neat.nn.RecurrentNetwork.create(genome, config)

# Usage is the same for both:
output = net.activate([input1, input2, ...])
# Returns a list with output values
```

> ⚠️ **Note:** To use a recurrent network, set `feed_forward = False` in the config file.

---

## 💾 Saving and Loading

### Save the best genome

```python
import pickle

# After finding the winner:
with open('winner.pkl', 'wb') as f:
    pickle.dump(winner, f)
```

### Load and use later

```python
import pickle
import neat

with open('winner.pkl', 'rb') as f:
    winner = pickle.load(f)

# You still need the config to build the network
config = neat.Config(...)
net = neat.nn.FeedForwardNetwork.create(winner, config)
output = net.activate([0, 1])
```

### Automatic checkpoints (saves the entire population)

```python
# Add to your reporters before running:
p.add_reporter(neat.Checkpointer(generation_interval=10))
# Saves every 10 generations as: neat-checkpoint-N

# To restore from a checkpoint:
p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-42')
p.run(eval_genomes, 10)  # continues from where it left off
```

---

## 📊 Visualizing Statistics

```python
import matplotlib.pyplot as plt

# stats was created with: stats = neat.StatisticsReporter()

# Fitness over generations
generation = range(len(stats.most_fit_genomes))
best_fitness = [g.fitness for g in stats.most_fit_genomes]

plt.plot(generation, best_fitness)
plt.xlabel('Generation')
plt.ylabel('Best Fitness')
plt.title('Fitness Evolution')
plt.grid(True)
plt.show()

# Other useful data:
stats.best_genome()           # Best genome of all time
stats.best_unique_genomes(5)  # Top 5 unique genomes
stats.get_fitness_mean()      # Average fitness per generation
stats.get_fitness_stdev()     # Standard deviation per generation
```

---

## ⚡ Parallelizing Evaluation (for slow environments)

When evaluating a genome is costly (e.g., physics simulations), use parallel evaluation:

```python
import neat
import multiprocessing

def eval_single_genome(genome, config):
    """
    Evaluates ONE genome (unlike eval_genomes which evaluates all).
    Must return the fitness value.
    """
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    fitness = 0.0
    # ... your logic here ...
    return fitness


def run(config_file):
    config = neat.Config(...)
    p = neat.Population(config)

    # Number of parallel processes
    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_single_genome)

    # Use pe.evaluate instead of eval_genomes directly
    winner = p.run(pe.evaluate, 50)
```

---

## 🐛 Common Errors and Solutions

| Error | Cause | Solution |
|-------|-------|----------|
| `KeyError: 'NEAT'` | Malformed config file | Check that sections `[NEAT]`, `[DefaultGenome]`, etc. exist |
| Fitness never improves | Wrong fitness function | Check that you're correctly rewarding/penalizing |
| `AttributeError: fitness is None` | Forgot to assign `genome.fitness` | Every genome must receive a fitness value |
| Evolution is very slow | `pop_size` too low | Increase population size (cost: more time per generation) |
| Network grows too large | `node_add_prob` too high | Reduce node/connection addition probabilities |

---

## 📋 Checklist for a New Project

```
[ ] Define: how many inputs and outputs does my network need?
[ ] Create config-feedforward.txt with correct num_inputs and num_outputs
[ ] Implement eval_genomes() with a clear fitness logic
[ ] Set a realistic fitness_threshold
[ ] Run with a small pop_size first to test
[ ] Add StdOutReporter to monitor progress
[ ] Save the winner with pickle at the end
```

---

## 🔗 Resources to Go Further

- 📚 [Official documentation](https://neat-python.readthedocs.io/)
- 📄 [Original NEAT paper (2002)](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
- 🎮 Classic examples: XOR, Cart-Pole, Flappy Bird, Atari games
- 🔬 Advanced variant: **HyperNEAT** (uses networks to generate networks — the *Inception* of neuroevolution)

---

> 💡 **Final thought:** NEAT isn't the fastest algorithm for every ML problem. But it shines when you **don't know the ideal architecture**, when the environment is **non-differentiable** (no gradient), or when you want **emergent behavior**. Here's something to reflect on: *in which problem from your own work would evolution make more sense than backpropagation?* 🤔
