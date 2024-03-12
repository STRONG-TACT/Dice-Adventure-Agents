# Dice Adventure Human-Machine Teaming Challenge
Welcome! This repository contains skeleton code to develop and train agents to play Dice Adventure (DA), developed 
at Carnegie Mellon University and Georgia Institute of Technology. The README serves as documentation for DA, how to
use the code in this repository, and information on the challenge itself.

- [Challenge Documentation](#challenge-docu)
- [Dice Adventure Documentation](#dice-adventure-documentation)
- [How to use code](#how-to-use-code)


## Dice Adventure Documentation
Dice Adventure is a 3D, turn-based, dungeon crawler game created at Carnegie Mellon University by [Dr. Erik Harpstead's](http://www.erikharpstead.net/) 
game design team in support of an Army Research Lab project on human-machine teaming (HMT). [Dr. Christopher MacLellan's](https://chrismaclellan.com/) 
Teachable AI Lab (TAIL) at the Georgia Institute of Technology supports the artificial intelligence development aspect 
of the project. DA was designed to explore paradigms in HMT related to the Army mission (*maybe link to HMT paradigms? Is that open info or FOUO?*). As such, DA's design integrates
concepts such as denied communication environments, *thing2*, and *thing3*.  

You can find a tutorial of how DA works [here](https://www.com). (~update~)  
## Challenge Documentation
In order to conduct research into some of the human-machine teaming

(*Insert link to challenge rules?*)

## How to use code
Although Dice Adventure (DA) was implemented in Unity and your agents will ultimately interface with the Unity version, 
this repository also implements a local, Python version of DA that users can use to develop agents. We developed the local
version as a way to train RL agents that required near clock-time feedback for millions of time steps. Please follow
these steps to set up your environment to use both the local and Unity versions of the game.

### Install required packages
Install the required packages using the `requirements.txt` file in order for DA to run. You may add additional packages
needed for your agents to this file.

```sh
  pip install -r requirements.txt
```

### Run the sample code
In the `examples/random_agent/` folder, there is a sample agent that plays as all three characters in the game and simply 
takes random actions. By default, it uses the Unity server to play (for which you will need the Unity version of the game
running), but you can change this at the top of the `play_game.py` file. 

```sh
  python play_game.py
```

### Develop your agents
We provide template code in the `agent_template/` folder. You should implement your agents within the `DiceAdventureAgent`
class provided in the `agent.py` file. This allows us to maintain a common API interface for all submitted agents.

The `init()` function should load in your agent using a relative filepath (if it needs to be loaded) and the `take_action()`
function should return an action to take in the game given the current game state and a list of game actions. Note that 
the return should be an option from the list of game actions, so if, for instance, your model enumerates the actions with
numbers, the conversion to a string should happen before the action is returned.

A custom Gymnasium environment ([good reference](https://blog.paperspace.com/creating-custom-environments-openai-gym/)) 
has been provided for convenience. This environment interfaces with both the local and Unity versions of the game via 
the `server` class declaration parameter and can be used to avoid setting up your own connection to both versions of the game.
No matter the type of agent you develop, you can add additional functions to the environment as needed to support training.
Each function is documented in the file.

### Agent Submission
Agents should be submitted at [this link]() (~update~) before the deadline. In your submission, please include a zip file 
of your `agent.py` file containing your agent implemented in the `DiceAdventureAgent` class, the `dice_adventure_python_env.py`
file, and any other model files needed to load in your agent. 