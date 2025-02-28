# Dice Adventure Human-Machine Teaming Challenge
Welcome! This repository contains skeleton code to develop and train agents to play Dice Adventure (DA), developed 
at Carnegie Mellon University and Georgia Institute of Technology. This README describes how to use the code. Please review the [changelog](https://github.com/STRONG-TACT/Dice-Adventure-Agents/blob/main/changelog.md) for any updates made to the code files.

- [Dice Adventure Documentation](#dice-adventure-documentation)
- [How to use code](#how-to-use-code)
  - [Install required packages](#install-required-packages)
  - [Run the sample code](#run-the-sample-code)
  - [Develop your agents](#develop-your-agents)
  - [Code Submission](#code-submission)


## Dice Adventure Documentation
Dice Adventure is a 3D, turn-based, dungeon crawler game created at Carnegie Mellon University by [Dr. Erik Harpstead's](http://www.erikharpstead.net/) 
game design team in support of an Army Research Lab project on human-machine teaming (HMT). [Dr. Christopher MacLellan's](https://chrismaclellan.com/) 
Teachable AI Lab (TAIL) at the Georgia Institute of Technology supports the artificial intelligence development aspect 
of the project. DA was designed to explore paradigms in HMT. 

You can find a complete game documentation on the [competition portal](https://strong-tact.github.io/).

## Code Overview
This repository provides five main files for you to develop and submit your agents. You can find these in the `agent_template/` and `game/` folders: 

- `agent.py`: A `DiceAdventureAgent` template class that you will use to implement your agent;
- `dice_adventure_gym_env.py`: An OpenAI Gym interface that lets you connect to the Dice Adventure Unity game and develop RL agents or other agent types;
- `submission_info.json`: A submission JSON file that requests basic information about your submission;
- ``: A Windows local build of Dice Adventure; and
- ``: A Mac local build of Dice Adventure

*Note: We tested this code in Python 3.9.0. Please see the end of this README for full list of software versions.*

### Install required packages
Install the required packages using the `requirements.txt` file in order for DA to run. You may add additional packages
needed for your agents to this file. Please note that we specified package versions to maintain consistency across this
project. If you encounter errors installing any of these versions, please [reach out to our team](#contact-information) for a resolution. 

```sh
  pip install -r requirements.txt
```

*Note: Not all packages may be necessary.*

### Run the sample code
In the `examples/random_agent/` folder, there is a sample agent that plays as all three characters in the game and simply 
takes random actions. 

```sh
  python play_game.py
```

The sample code is commented and can be used as a reference structure for training/testing agents. 

### Develop your agents
We provide template code in the `agent_template/` folder. You should implement your agents within the `DiceAdventureAgent`
class provided in the `agent.py` file. This allows us to maintain a common API interface for all submitted agents.

The `init()` function should load in your agent using a relative filepath (if it needs to be loaded) and the `take_action()`
function should return an action to take in the game given the current game state and a list of game actions. Note that 
the return should be an option from the list of game actions, so if, for instance, your model enumerates the actions with
numbers, the conversion to a string should happen before the action is returned.

A custom Gymnasium environment ([good reference](https://blog.paperspace.com/creating-custom-environments-openai-gym/)) 
has been provided for convenience in the file `dice_adventure_python_env.py`. This environment interfaces with both the local and Unity versions of the game via 
the `server` class declaration parameter and can be used to avoid setting up your own connection to both versions of the game.
No matter the type of agent you develop, you can add additional functions to the environment as needed to support training.
Each function is documented in the file.

### Code Submission
Agents should be submitted on the [competition portal](https://strong-tact.github.io) before the deadline. 
In your submission, please include a zipped folder containing:
  - your `agent.py` file containing your agent implemented in the `DiceAdventureAgent` class
  - your `dice_adventure_python_env.py` file
  - the `submission_info.json` file containing your name and your agent's name
  - a `requirements.txt` file containing any packages your model needs to run
  - any other model files needed to load in your agent

### Contact Information
Please contact any of us if you have any questions or run into any issues!

| Name       | Position    | Contact              | 
|------------|-------------|----------------------|
| Qiao Zhang | PhD Student | qz99(at)gatech.edu |

### Versions
| Tool/Software  | Version |
|----------------|---------|
| Python         | 3.9.0   |
| Unity Hub      | 3.5.1   |
| Unity Editor   | 2020.3.32f1 |

