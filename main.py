# !/usr/bin/env python3
# usage: main.py [-h] [--name NAME]

import os
import sys
import asyncio
import random

from typing import Optional, List

sys.path.append(os.path.join(os.path.abspath("."), "among-agents"))

import argparse
import datetime
import subprocess

from amongagents.envs.configs.agent_config import ALL_LLM
from amongagents.envs.configs.game_config import FIVE_MEMBER_GAME, SEVEN_MEMBER_GAME, FIVE_MEMBER_GAME
from amongagents.envs.configs.map_config import map_coords
from amongagents.envs.game import AmongUs
from amongagents.UI.MapUI import MapUI
from dotenv import load_dotenv

from utils import setup_experiment

ROOT_PATH = os.path.abspath(".")
LOGS_PATH = os.path.join(ROOT_PATH, "expt-logs")
ASSETS_PATH = os.path.join(ROOT_PATH, "among-agents", "amongagents", "assets")
BLANK_MAP_IMAGE = os.path.join(ASSETS_PATH, "blankmap.png")

load_dotenv()

DATE = datetime.datetime.now().strftime("%Y-%m-%d")
COMMIT_HASH = (
    subprocess.check_output(["git", "rev-parse", "HEAD"]).strip().decode("utf-8")
)

BIG_LIST_OF_MODELS: List[str] = [
    "meta-llama/llama-3.3-70b-instruct",
    "qwen/qwen3-30b-a3b-instruct-2507",
    "mistralai/mistral-small-3.2-24b-instruct",
    "microsoft/phi-4"
]

ARGS = {
    "game_config": SEVEN_MEMBER_GAME,
    "include_human": False,
    "test": False,
    "personality": False,
    "agent_config": {
        "Impostor": "LLM",
        "Crewmate": "LLM",
        "IMPOSTOR_LLM_CHOICES": BIG_LIST_OF_MODELS,
        "CREWMATE_LLM_CHOICES": BIG_LIST_OF_MODELS,
    },
    "UI": False,
}

async def multiple_games(experiment_name=None, num_games=1, rate_limit=50):
    experiment_name = setup_experiment(experiment_name, LOGS_PATH, DATE, COMMIT_HASH, ARGS)
    ui = MapUI(BLANK_MAP_IMAGE, map_coords, debug=False) if ARGS["UI"] else None
    with open(os.path.join(os.environ["EXPERIMENT_PATH"], "experiment-details.txt"), "a") as experiment_file:
        experiment_file.write(f"\nExperiment args: {ARGS}\n")

    semaphore = asyncio.Semaphore(rate_limit)

    async def run_limited_game(game_index):
        async with semaphore:
            if ARGS.get("tournament_style") == "1on1":
                # Randomly select one model for each role for this specific game
                game_config = ARGS["agent_config"].copy()
                game_config["CREWMATE_LLM_CHOICES"] = [random.choice(BIG_LIST_OF_MODELS)]
                game_config["IMPOSTOR_LLM_CHOICES"] = [random.choice(BIG_LIST_OF_MODELS)]
            else:
                game_config = ARGS["agent_config"]
                
            game = AmongUs(
                game_config=ARGS["game_config"],
                include_human=ARGS["include_human"],
                test=ARGS["test"],
                personality=ARGS["personality"],
                agent_config=game_config,
                UI=ui,
                game_index=game_index,
            )
            await game.run_game()

    tasks = [run_limited_game(i) for i in range(1, num_games+1)]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run an AmongUs experiment.")
    parser.add_argument("--name", type=str, default=None, help="Optional name for the experiment.")
    parser.add_argument("--num_games", type=int, default=2, help="Number of games to run.")
    parser.add_argument("--display_ui", type=bool, default=False, help="Display UI.")
    parser.add_argument("--crewmate_llm", type=str, default=None, help="Crewmate LLM model.")
    parser.add_argument("--impostor_llm", type=str, default=None, help="Impostor LLM model.")
    parser.add_argument("--streamlit", type=bool, default=False, help="Streamlit.")
    parser.add_argument("--memory", type=str, default="text", choices=["text", "kg"],
                        help="Agent memory: 'text' = baseline self-written Condensed Memory, 'kg' = knowledge graph.")
    parser.add_argument("--tournament_style", type=str, default="random", help="random or 1on1.")
    args = parser.parse_args()
    if args.num_games > 1 or args.display_ui == False:
        ARGS["UI"] = False
    if args.crewmate_llm:
        ARGS["agent_config"]["CREWMATE_LLM_CHOICES"] = [args.crewmate_llm]
    if args.impostor_llm:
        ARGS["agent_config"]["IMPOSTOR_LLM_CHOICES"] = [args.impostor_llm]
    ARGS["tournament_style"] = args.tournament_style
    ARGS["agent_config"]["MEMORY"] = args.memory
    asyncio.run(multiple_games(experiment_name=args.name, num_games=args.num_games))