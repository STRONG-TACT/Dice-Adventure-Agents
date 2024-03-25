Any changes to the code in this repository will be tracked and communicated here in this changelog. 

*3/25/2025*
- Implemented two additional ways to pull the state from the environment. When using the DiceAdventurePythonEnv `get_state()`
  function, users can pass in an option `player` and `version` parameter. `player` can be one of {Dwarf, Giant, Human}
  and `version` can be one of {full, character, fow}. If no values are passed, the class defaults, or values passed in during
  class declaration are used. Please see the function declaration for documentation of these parameters.
- Changed `version` parameter option in `dice_adventure_python_env.get_state()` from "character" to "player". The options
  for this parameter are now {full, player, fow}.