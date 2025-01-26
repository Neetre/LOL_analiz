# config.py
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).parent.parent


RIOT_API_CONFIG = {
    "base_url": "https://{}.api.riotgames.com",
    "ddragon_url": "https://ddragon.leagueoflegends.com/cdn/14.14.1/data/en_US/champion.json",
    "regions": {
        "servers": ["americas", "asia", "europe"],
        "server_to_region": {"kr": "kr", "euw1": "europe", "na1": "americas"}
    }
}

PATHS = {
    "data": BASE_DIR / "data",
    "logs": BASE_DIR / "logs"
}

PATHS["data"].mkdir(exist_ok=True)
PATHS["logs"].mkdir(exist_ok=True)