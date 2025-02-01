import os
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import json
from fastapi.responses import JSONResponse, HTMLResponse
import logging


app = FastAPI(
    title="League of Legends Game Predictor API",
    description="API for predicting League of Legends game outcomes and analyzing champion statistics",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")

class Champion(BaseModel):
    name: str


class Team(BaseModel):
    champions: List[str]


class GameData(BaseModel):
    t1_team: Dict[str, str]  # {"t1_champ1": "ChampName", ...}
    t2_team: Dict[str, str]  # {"t2_champ1": "ChampName", ...}
    game_stats: Dict[str, int]


class PredictionResponse(BaseModel):
    team1_win_probability: float
    team2_win_probability: float
    predicted_winner: int
    model_accuracy: float


class WinrateResponse(BaseModel):
    champion: str
    winrate: float
    total_games: int


class User(BaseModel):
    username: str
    password: str


class LoginRequest(BaseModel):
    username: str
    password: str


MODEL = None
CHAMPION_DATA = None
CHAMPION_DICT = {}  # {champion_name: champion_id}
TRAINING_DATA = None
MODEL_ACCURACY = 0.0


def load_data():
    """Initialize the model and load required data"""
    global MODEL, CHAMPION_DATA, CHAMPION_DICT, TRAINING_DATA
    
    try:
        TRAINING_DATA = pd.read_csv('../data/game_DIAMOND.csv')
        with open('../data/champion_info.json', 'r') as f:
            CHAMPION_DATA = json.load(f)

        for champ_name, champ_data in CHAMPION_DATA['data'].items():
            CHAMPION_DICT[champ_name.lower()] = int(champ_data['id'])

    except FileNotFoundError as e:
        raise RuntimeError(f"Required data files not found: {str(e)}")

    train_model()


def train_model():
    global MODEL, TRAINING_DATA, MODEL_ACCURACY
    
    processed_data = clean_df(TRAINING_DATA, start=False)
    X = processed_data.drop('winner', axis=1)
    y = processed_data['winner']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    MODEL = RandomForestClassifier(n_jobs=-1)
    MODEL.fit(X_train, y_train)
    
    MODEL_ACCURACY = MODEL.score(X_test, y_test)
    return MODEL_ACCURACY


def clean_df(df: pd.DataFrame, start: bool = False, game_stats: dict = None, is_prediction: bool = False) -> pd.DataFrame:
    """Clean and process the DataFrame"""
    t1 = [f"t1_champ{i}" for i in range(1, 6)]
    t2 = [f"t2_champ{i}" for i in range(1, 6)]
    
    additional = ["winner"] if not is_prediction else []
    if not start and game_stats:
        if game_stats.get('first_blood') is not None:
            additional.append('firstBlood')
        if game_stats.get('first_dragon') is not None:
            additional.append('firstDragon')
        if game_stats.get('first_tower') is not None:
            additional.append('firstTower')
        if game_stats.get('first_inhibitor') is not None:
            additional.append('firstInhibitor')
        if game_stats.get('first_baron') is not None:
            additional.append('firstBaron')
        if game_stats.get('first_rift_herald') is not None:
            additional.append('firstRiftHerald')
    
    columns_to_keep = t1 + t2 + additional
    df = df[columns_to_keep]

    dummies = [pd.get_dummies(df[col], prefix=f"t{i//5+1}") for i, col in enumerate(t1 + t2)]
    df = pd.concat([df] + dummies, axis=1)
    df = df.drop(t1 + t2, axis=1)
    df = df.fillna(0)
    
    return df


def prepare_prediction_data(game_data: GameData) -> pd.DataFrame:
    """Prepare data for prediction based on the new structure."""
    t1_champs = []
    for i in range(5):
        champ_name = game_data.t1_team[f"t1_champ{i+1}"]
        champ_id = CHAMPION_DICT.get(champ_name.lower())
        if champ_id is None:
            raise ValueError(f"Champion {champ_name} not found")
        t1_champs.append(champ_id)
    
    t2_champs = []
    for i in range(5):
        champ_name = game_data.t2_team[f"t2_champ{i+1}"]
        champ_id = CHAMPION_DICT.get(champ_name.lower())
        if champ_id is None:
            raise ValueError(f"Champion {champ_name} not found")
        t2_champs.append(champ_id)
    
    data = {
        **{f't1_champ{i+1}': champ_id for i, champ_id in enumerate(t1_champs)},
        **{f't2_champ{i+1}': champ_id for i, champ_id in enumerate(t2_champs)}
    }

    game_stats = game_data.game_stats
    if game_stats:
        data.update({
            'firstBlood': game_stats.get('firstBlood', 0),
            'firstDragon': game_stats.get('firstDragon', 0),
            'firstTower': game_stats.get('firstTower', 0),
            'firstInhibitor': game_stats.get('firstInhibitor', 0),
            'firstBaron': game_stats.get('firstBaron', 0),
            'firstRiftHerald': game_stats.get('firstRiftHerald', 0)
        })

    df = pd.DataFrame([data])
    return clean_df(df, start=all(v == 0 for v in game_stats.values()), 
                   game_stats=game_stats, is_prediction=True)


@app.on_event("startup")
async def startup_event():
    """Load data and train model on startup"""
    load_data()

'''
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render the home page"""
    return templates.TemplateResponse("index.html", {"request": request})
'''


@app.post("/predict", response_model=PredictionResponse)
async def predict_game(game_data: GameData):
    """
    Predict the outcome of a game based on team compositions and optional game statistics
    """
    if MODEL is None:
        raise HTTPException(status_code=500, detail="Model not initialized")

    try:
        predict_data = prepare_prediction_data(game_data)

        missing_cols = set(MODEL.feature_names_in_) - set(predict_data.columns)
        missing_df = pd.DataFrame(0, index=predict_data.index, columns=list(missing_cols))
        predict_data = pd.concat([predict_data, missing_df], axis=1)
        predict_data = predict_data[MODEL.feature_names_in_]
        
        probabilities = MODEL.predict_proba(predict_data)[0]
        prediction = MODEL.predict(predict_data)[0]
        
        return {
            "team1_win_probability": float(probabilities[0]),
            "team2_win_probability": float(probabilities[1]),
            "predicted_winner": int(prediction),
            "model_accuracy": float(MODEL_ACCURACY)  # Use stored accuracy
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/champion_names")
async def get_champion_names():
    """Get the list of all champion names"""
    if CHAMPION_DATA is None:
        raise HTTPException(status_code=500, detail="Champion data not initialized")
    
    try:
        champion_names = list(CHAMPION_DATA["data"].keys())
        return JSONResponse(content=champion_names)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/champion/{champion_name}/winrate", response_model=WinrateResponse)
async def get_champion_winrate(champion_name: str):
    """
    Get the winrate statistics for a specific champion
    """
    if TRAINING_DATA is None:
        raise HTTPException(status_code=500, detail="Training data not initialized")
    
    try:
        champ_id = CHAMPION_DICT.get(champion_name.lower())
        if champ_id is None:
            raise HTTPException(status_code=404, detail=f"Champion {champion_name} not found")

        t1_cols = [f't1_champ{i}' for i in range(1, 6)]
        t2_cols = [f't2_champ{i}' for i in range(1, 6)]

        t1_matches = TRAINING_DATA[TRAINING_DATA[t1_cols].eq(champ_id).any(axis=1)]
        wins1 = len(t1_matches[t1_matches['winner'] == 1])
        losses1 = len(t1_matches[t1_matches['winner'] == 2])

        t2_matches = TRAINING_DATA[TRAINING_DATA[t2_cols].eq(champ_id).any(axis=1)]
        wins2 = len(t2_matches[t2_matches['winner'] == 2])
        losses2 = len(t2_matches[t2_matches['winner'] == 1])
        
        total_games = wins1 + wins2 + losses1 + losses2
        if total_games == 0:
            raise HTTPException(status_code=404, detail=f"No games found for champion {champion_name}")
            
        winrate = (wins1 + wins2) / total_games
        return WinrateResponse(
            champion=champion_name,
            winrate=float(winrate),
            total_games=total_games
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/predict/random")
async def random_prediction():
    """Generate a random game and predict the outcome"""
    if CHAMPION_DATA is None:
        raise HTTPException(status_code=500, detail="Champion data not initialized")
    
    try:
        champions = list(CHAMPION_DATA["data"].keys())
        team1_champs = np.random.choice(champions, 5)
        team2_champs = np.random.choice(champions, 5)

        t1_team = {f"t1_champ{i+1}": champ for i, champ in enumerate(team1_champs)}
        t2_team = {f"t2_champ{i+1}": champ for i, champ in enumerate(team2_champs)}

        game_data = GameData(
            t1_team=t1_team,
            t2_team=t2_team,
            game_stats={
                "firstBlood": 0,
                "firstDragon": 0,
                "firstTower": 0,
                "firstInhibitor": 0,
                "firstBaron": 0,
                "firstRiftHerald": 0
            }
        )

        data = await predict_game(game_data)
        return {
            "teams": {
                "team1": team1_champs.tolist(),
                "team2": team2_champs.tolist()
            },
            "prediction": {
                "team1_win_probability": data["team1_win_probability"],
                "team2_win_probability": data["team2_win_probability"],
                "predicted_winner": data["predicted_winner"],
                "model_accuracy": data["model_accuracy"]
            }
        }
    except Exception as e:
        print(e)
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/health")
async def health_check():
    """
    Check if the API is healthy and model is loaded
    """
    return {"status": "healthy", "model_loaded": MODEL is not None}