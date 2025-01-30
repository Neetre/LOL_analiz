import requests
from requests.exceptions import HTTPError, Timeout, TooManyRedirects, RequestException
import json
import time
import os
import logging
from typing import Optional, Dict, List, Any
from dataclasses import dataclass
from datetime import datetime
import random
from pathlib import Path
from scrap import get_riot_ids
from dotenv import load_dotenv
load_dotenv()


def get_season():
    link = "https://ddragon.leagueoflegends.com/api/versions.json"
    response = requests.get(link)
    data = response.json()
    version = data[0]
    link = f"http://ddragon.leagueoflegends.com/cdn/{version}/data/en_US/champion.json"
    return link


@dataclass
class RiotConfig:
    api_key: str
    base_url: str = "https://{region}.api.riotgames.com"
    ddragon_url: str = get_season()
    max_retries: int = 5
    base_delay: float = 1.2
    jitter: float = 0.5
    regions: Dict[str, List[str]] = None

    def __post_init__(self):
        self.regions = {
            "americas": ["na1", "br1", "la1", "la2"],
            "asia": ["kr", "jp1"],
            "europe": ["euw1", "eun1", "tr1", "ru"]
        }

class RateLimiter:
    def __init__(self, base_delay: float = 1.2, jitter: float = 0.5):
        self.base_delay = base_delay
        self.jitter = jitter
        self.last_request_time = 0

    def wait(self):
        """Implements intelligent waiting with jitter to prevent rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.base_delay:
            jitter_amount = random.uniform(0, self.jitter)
            sleep_time = self.base_delay - time_since_last + jitter_amount
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()

class RiotAPIClient:
    def __init__(self, config: RiotConfig):
        self.config = config
        self.rate_limiter = RateLimiter(config.base_delay, config.jitter)
        self.logger = self._setup_logger()
        self.session = requests.Session()
        self.session.headers.update({
            "X-Riot-Token": config.api_key,
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })

    def _setup_logger(self) -> logging.Logger:
        logger = logging.getLogger("RiotAPI")
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.FileHandler("riot_api.log")
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger

    def _make_request(self, endpoint: str, region: str, params: Dict = None) -> Optional[Dict]:
        """Make an HTTP request with retry logic and rate limiting"""
        url = self.config.base_url.format(region=region) + endpoint
        
        for attempt in range(self.config.max_retries):
            try:
                self.rate_limiter.wait()
                response = self.session.get(url, params=params or {})
                response.raise_for_status()
                return response.json()
            
            except HTTPError as e:
                if response.status_code == 429:
                    retry_after = int(response.headers.get('Retry-After', 5))
                    self.logger.warning(f"Rate limit exceeded. Waiting {retry_after} seconds.")
                    time.sleep(retry_after)
                    continue
                
                self.logger.error(f"HTTP error {response.status_code}: {str(e)}")
                if 500 <= response.status_code < 600:
                    time.sleep(2 ** attempt)
                    continue
                return None
            
            except Exception as e:
                self.logger.error(f"Request failed: {str(e)}")
                time.sleep(2 ** attempt)
        
        return None

    def get_puuid(self, region: str, username: str, tag: str) -> Optional[str]:
        """Get PUUID for a given Riot ID"""
        endpoint = f"/riot/account/v1/accounts/by-riot-id/{username}/{tag}"
        response = self._make_request(endpoint, region)
        return response.get("puuid") if response else None

    def get_match_history(self, region: str, puuid: str, count: int = 20) -> Optional[List[str]]:
        """Get match history for a player"""
        endpoint = f"/lol/match/v5/matches/by-puuid/{puuid}/ids"
        return self._make_request(endpoint, region, params={"count": count})

    def get_match_details(self, region: str, match_id: str) -> Optional[Dict]:
        """Get detailed match information"""
        endpoint = f"/lol/match/v5/matches/{match_id}"
        return self._make_request(endpoint, region)

class MatchDataProcessor:
    def __init__(self, champion_data: Dict[str, Any]):
        self.champion_data = champion_data

    def process_match(self, match_data: Dict) -> Optional[List]:
        """Process match data into a format suitable for CSV storage"""
        try:
            info = match_data['info']
            teams = info['teams']

            match_info = [
                info['gameId'],
                info['gameCreation'],
                info['gameDuration'],
                info['gameVersion'].split('.')[0],
                1 if teams[0]['win'] else 2
            ]

            objectives = teams[0]['objectives']
            first_objectives = [
                objectives[obj]['first']
                for obj in ['champion', 'tower', 'inhibitor', 'baron', 'dragon', 'riftHerald']
            ]
            match_info.extend([1 if obj else 2 for obj in first_objectives])

            champions = self._process_champions(info['participants'])
            if not champions:
                return None
                
            match_info.extend(champions)
            return match_info
            
        except KeyError as e:
            logging.error(f"Error processing match data: {e}")
            return None

    def _process_champions(self, participants: List[Dict]) -> Optional[List[int]]:
        """Process champion information from match participants"""
        position_map = {"TOP": 1, "JUNGLE": 2, "MIDDLE": 3, "BOTTOM": 4, "UTILITY": 5}
        
        try:
            champions = sorted(
                [(p['championName'], position_map[p['individualPosition']])
                 for p in participants],
                key=lambda x: (x[1], x[0])
            )
            
            return [self.champion_data['data'][champ]['id'] 
                   for champ, _ in champions 
                   if champ in self.champion_data['data']]
        except KeyError:
            return None

class DataCollector:
    def __init__(self, config: RiotConfig, output_dir: Path):
        self.config = config
        self.client = RiotAPIClient(config)
        self.output_dir = output_dir
        self.champion_processor = self._init_champion_processor()

    def _init_champion_processor(self) -> MatchDataProcessor:
        """Initialize the champion processor with champion data"""
        response = requests.get(self.config.ddragon_url)
        champion_data = response.json()
        return MatchDataProcessor(self._process_champion_data(champion_data))

    def _process_champion_data(self, raw_data: Dict) -> Dict:
        """Process raw champion data into a more usable format"""
        processed = {
            "type": "champion",
            "version": raw_data['version'],
            "data": {}
        }
        
        for i, (key, champ) in enumerate(raw_data['data'].items(), start=1):
            name = "FiddleSticks" if champ['name'] == "Fiddlesticks" else champ['name']
            key = "FiddleSticks" if key == "Fiddlesticks" else key
            
            processed['data'][key] = {
                "title": champ['title'],
                "id": i,
                "key": key,
                "name": name
            }

        with open("../data/champion_info.json", "w") as f:
            json.dump(processed, f, indent=4)
        
        return processed

    def collect_data(self, region: str, tier: str, pages: List[str], matches_per_player: int = 20):
        """Main data collection method"""
        csv_path = self.output_dir / f"game_{tier}.csv"

        if not csv_path.exists():
            self._init_csv(csv_path)

        for page in pages:
            self._collect_page_data(region, tier, page, matches_per_player, csv_path)

    def _init_csv(self, csv_path: Path):
        """Initialize a CSV file with headers"""
        with open(csv_path, 'w') as f:
            f.write("gameId,gameCreation,gameDuration,gameVersion,winner,firstBlood,firstTower,firstInhibitor,firstBaron,firstDragon,firstRiftHerald,t1_champ1,t1_champ2,t1_champ3,t1_champ4,t1_champ5,t2_champ1,t2_champ2,t2_champ3,t2_champ4,t2_champ5\n")

    def _append_to_csv(self, csv_path: Path, data: List):
        """Append data to a CSV file"""
        with open(csv_path, 'a') as f:
            f.write(f"{','.join(map(str, data))}\n")

    def _collect_page_data(self, region: str, tier: str, page: str, matches_per_player: int, csv_path: Path):
        """Collect data for a specific page of players"""
        riot_ids = get_riot_ids(region, tier, page)

        for username, tag in riot_ids:
            puuid = self.client.get_puuid(region, username, tag)
            if not puuid:
                continue
                
            matches = self.client.get_match_history(region, puuid, matches_per_player)
            if not matches:
                continue
                
            self._process_matches(matches, region, csv_path)

    def _process_matches(self, matches: List[str], region: str, csv_path: Path):
        """Process a list of matches and save to CSV"""
        for match_id in matches:
            match_data = self.client.get_match_details(region, match_id)
            if not match_data:
                continue
                
            processed_data = self.champion_processor.process_match(match_data)
            if processed_data:
                self._append_to_csv(csv_path, processed_data)


def main():
    config = RiotConfig(
        api_key=os.getenv("RIOT_KEY"),
        base_delay=1.2,
        jitter=0.5
    )

    output_dir = Path("../data")
    output_dir.mkdir(exist_ok=True)

    collector = DataCollector(config, output_dir)

    collector.collect_data(
        region="europe",
        tier="EMERALD",
        pages=["1", "2", "3"],
        matches_per_player=20
    )

if __name__ == "__main__":
    main()