# AE4350 Project: Reinforcement Learning-Based Multi-Agent Collision Avoidance System
This repository contains the code developed for the course AE4350, as part of the Master's in Aerospace Engineering degree at TU Delft. The project aims to design and implement a reinforcement learning-based multi-agent collision avoidance system for air traffic management. The goal is to showcase understanding of reinforcement learning methods within a distributed learning systems in managing aircraft separation and preventing collisions in a free-flight environment.

## Project Goal
The objective of this project is to create a reinforcement learning (RL) model capable of navigating multiple aircraft within a shared airspace while avoiding collisions. The system is designed to be decentralized, where each aircraft (agent) independently makes decisions based on its environment and learned policies.
The project makes use of multiple agents that use and update a shared policy.

## Project Structure
```text
.
├── src/
│   ├── edge.py                 # Helper class to create start and end locations along environment edges
│   ├── ma_airplane.py          # Main module defining the Aircraft class
│   ├── ma_environment.py       # Main module defining the Environment class wherein each Aircraft is contained
│   └── visualizer.py           # Pygame visualizer for rendering the environment and agent movements
├── utils/
│   ├── normalizers.py          # Module containing normalization functions
│   └── units.py                # Module containing unit conversion functions
├── main_ma.py                  # Main script to run train and evaluation functions
├── README.md                   # Project documentation
├── pyproject.toml              # Poetry configuration file for managing dependencies
└── .gitignore                  # Git ignore file to exclude unnecessary files
```

## Cloning the Repository

To clone the repository and install the necessary dependencies using Poetry, follow these steps:

### Clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```

### Install dependencies using Poetry:
If you don't have `poetry` already, download and install it from [here](https://python-poetry.org/docs/#installation).
```bash
poetry install
```
