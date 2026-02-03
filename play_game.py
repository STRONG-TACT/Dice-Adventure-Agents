import argparse
import os
from pathlib import Path

from agent_template.dice_adventure_gym_env import DiceAdventureGymEnv
from agent_template.dice_adventure_gym_env import get_player_id
from agent_template.agent import DiceAdventureAgent


def build_agent(character: str, model_path: str | None) -> tuple[str, DiceAdventureAgent]:
    """
    Helper to create an agent tuple (character, DiceAdventureAgent) that env.play expects.
    """
    if model_path:
        expanded_path = Path(model_path).expanduser()
        if not expanded_path.exists():
            print(f"[play_game] Warning: Model for {character} not found at {expanded_path}. Using random actions.")
            model_path = None
        else:
            model_path = str(expanded_path)
    else:
        print(f"[play_game] No model provided for {character}. Using random actions.")

    agent = DiceAdventureAgent(character, get_player_id(character), model_path=model_path)
    return character, agent


def parse_args():
    parser = argparse.ArgumentParser(description="Play Dice Adventure with DQN agents.")
    parser.add_argument(
        "--game",
        type=str,
        default="./DiceAdventure_MacOS_Build.app/Contents/MacOS/DiceAdventure"
        if os.name == "posix"
        else "/DiceAdventure.exe",
        help="Path to the Dice Adventure executable.",
    )
    parser.add_argument("--port", type=str, default="4649", help="WebSocket port to use.")
    parser.add_argument("--dwarf-model", type=str, default=None, help="Path to dwarf model weights.")
    parser.add_argument("--giant-model", type=str, default=None, help="Path to giant model weights.")
    parser.add_argument("--human-model", type=str, default=None, help="Path to human model weights.")
    parser.add_argument(
        "--model-dir",
        type=str,
        default="models",
        help="Directory containing model checkpoints (used if specific model paths not provided).",
    )
    return parser.parse_args()


def resolve_default_model(model_path: str | None, model_dir: str, character: str) -> str | None:
    """If no explicit path, try to use `<model_dir>/<character>_dqn_final.weights.h5`."""
    if model_path:
        return model_path

    candidate = Path(model_dir) / f"{character}_dqn_final.weights.h5"
    return str(candidate) if candidate.exists() else None


def main():
    args = parse_args()

    dwarf_model = resolve_default_model(args.dwarf_model, args.model_dir, "dwarf")
    giant_model = resolve_default_model(args.giant_model, args.model_dir, "giant")
    human_model = resolve_default_model(args.human_model, args.model_dir, "human")

    env = DiceAdventureGymEnv(port=args.port, game_executable_filepath=args.game)

    agents = [
        build_agent("dwarf", dwarf_model),
        build_agent("giant", giant_model),
        build_agent("human", human_model),
    ]

    env.play(agents)


if __name__ == "__main__":
    main()
