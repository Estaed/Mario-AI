# Mario AI: Reinforcement Learning in Super Mario

## Overview
This project was developed as part of my Bachelor's degree at Firat University, similar to the Doom AI project. It explores the application of reinforcement learning algorithms to train an AI agent to play Super Mario. The goal is to analyze how different algorithms perform in terms of training efficiency, reward accumulation, and gameplay effectiveness.

The research paper included in this repository is the same as the one used for the Doom AI project.
This project was developed as part of my Bachelor's degree at Firat University. It explores the application of reinforcement learning algorithms to train an AI agent to play Super Mario. The goal is to analyze how different algorithms perform in terms of training efficiency, reward accumulation, and gameplay effectiveness.

## Repository Contents
- **Jupyter Notebook**: Contains the full implementation. Run all cells in order to execute the experiments.
- **Training Logs**: Collected data from multiple training sessions.
- **Results & Analysis**: Performance comparison including graphs and insights.
- **Paper**: A detailed report explaining the methodology, results, and conclusions.

## Requirements
To run this project, install the necessary dependencies:
```bash
pip install gym[all] stable-baselines3 gym-super-mario-bros numpy matplotlib
```

## Running the Code
1. Clone the repository:
```bash
git clone https://github.com/Estaed/Mario-AI.git
cd Mario-AI
```
2. Open the Jupyter Notebook and run all cells in order.

## Results
- **PPO was more successful compared to DQN.** While PPO was able to finish Level 1 and pass some areas in Level 2, DQN couldn't even finish Level 1.
- **Performance Evaluation**: Different reinforcement learning algorithms were tested, and their effectiveness was compared.
- **Challenges**: Training stability, exploration strategies, and optimization techniques.

For detailed findings, refer to the included research paper.
- **Performance Evaluation**: Different reinforcement learning algorithms were tested, and their effectiveness was compared.
- **Challenges**: Training stability, exploration strategies, and optimization techniques.

For detailed findings, refer to the included research paper.

## Future Work
- Fine-tuning hyperparameters for better performance.
- Exploring alternative reinforcement learning techniques.
- Expanding the project to more complex Super Mario levels.

## Author
Tarık Bulut

For any questions or suggestions, feel free to open an issue or contact me!

