Any changes to the code in this repository will be tracked and communicated here in this changelog. 

*3/25/2024*
- **Get_State() Additions**: Implemented two additional ways to pull the state from the environment. When using the DiceAdventurePythonEnv `get_state()`
  function, users can pass in an optional `player` and `version` parameter. `player` can be one of {Dwarf, Giant, Human}
  and `version` can be one of {full, player, fow}. If no values are passed, the class defaults, or values passed in during
  Env class declaration are used. Please see the function declaration for documentation of these parameters.


*5/7/2024*
- **Phase Timer**: Phases are now limited by a timer to keep gameplay moving. Player pinning and action planning will 
  each expire after 30 seconds by default, after which any actions taken so far in that phase will be auto-submitted and 
  the next phase will automatically begin.
- **Player Lives**: Players now have 3 lives in addition to 3 health. When all 3 health is exhausted, the player loses a
  life, is removed from the board, and respawns after one full round has completed. When all lives are exhausted, the 
  player can no longer respawn, ending the game for all players.
- **New Object Codes**: In the Python version, player codes have been changed from `1S, 2S, and 3S to C1, C2, and C3` for 
  the Dwarf, Giant, and Human, respectively, and shrine codes have been changed from `1G, 2G, and 3G to K1, K2, and K3`
  (K for "key shrines") to follow the same pattern as other object codes.


*6/5/2024*
- **New Character Argument**: We added an argument to the DiceAdventureAgent class `character` so that the agent can
  know which character to submit actions for. This will be useful since agents may need to play as any character and may 
  have special logic for different players. 
- **Environment File Changes**:
  - We modified the `step()` function so that depending on the value of `self.train_mode`, the agent will learn 
    (train_mode==True) or simply play (train_mode==False). the interface to the step function remains the same.
  - We added two helper functions `get_actions()` that returns a list of all game actions and `get_player_names()` 
    that returns a list of all 3 character names in the game.
- **Example Agent**: We added an example agent `examples/production_agent` that implements the DiceAdventureAgent class
  and plays based on conditional if/elif/else rules. You should be able to modify the `play_game.py` file to reference
  this agent and have it play the game. One note is that based on its movement logic, it can get stuck behind walls so
  this is something a more robust agent would need to improve on.
