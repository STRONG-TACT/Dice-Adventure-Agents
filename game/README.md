# HMT-GAME-1-AGENTS
This repository implements a low fidelity version of HMT Game 1 (Dice Adventure), a custom OpenAI Gym environment for Dice Adventure, and training code for agents to connect and learn to play the game

## Install required packages
```sh
  pip install -r requirements.txt
```

## Run the PPO model to play Dice Adventure
```sh
  python train/train_agent.py
```

## In another terminal, start up the TensorBoard server to view training results
```sh
  tensorboard --logdir ./dice_adventure_tensorboard/
```
