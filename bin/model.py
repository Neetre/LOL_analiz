import os
import pandas as pd
import numpy as np
import json
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import yaml
import argparse

from utils.logger import logger


def elab_champs(df: pd.DataFrame, champ_data: dict) -> pd.DataFrame:
    for team in ["t1", "t2"]:
        for i in range(1, 6):
            col_id = f"{team}_champ{i}id"
            col_name = f"{team}_champ{i}"
            df[col_name] = df[col_id].astype(str).map(
                lambda x: champ_data["data"].get(x, {}).get("name", "Unknown")
            )
    return df

def clean_df(df, start=False, more=None, is_prediction=False):
    t1 = ["t1_champ1", "t1_champ2", "t1_champ3", "t1_champ4", "t1_champ5"]
    t2 = ["t2_champ1", "t2_champ2", "t2_champ3", "t2_champ4", "t2_champ5"]
    
    additional = ["winner"] if not is_prediction else []
    if not start:
        additional.append("firstBlood")
    if more is not None:
        additional.extend([
            'firstDragon' if more >= 1 else None,
            'firstTower' if more >= 2 else None,
            'firstInhibitor' if more >= 3 else None,
            'firstBaron' if more >= 4 else None,
            'firstRiftHerald' if more >= 5 else None
        ])
    additional = [col for col in additional if col is not None]
    
    columns_to_keep = t1 + t2 + additional
    df = df[columns_to_keep]

    dummies = [pd.get_dummies(df[col], prefix=f"t{i//5+1}") for i, col in enumerate(t1 + t2)]
    df = pd.concat([df] + dummies, axis=1)
    df = df.drop(t1 + t2, axis=1)
    df = df.fillna(0)
    
    return df


def extract_data(file: str, champ_data: dict) -> pd.DataFrame:
    with open(file, 'r') as f:
        data = yaml.safe_load(f)
    
    t1_champions = [data['t1_team'][key] for key in data['t1_team'] if data['t1_team'][key]]
    t2_champions = [data['t2_team'][key] for key in data['t2_team'] if data['t2_team'][key]]
    
    t1_champions = [champ_data['data'][champ]['id'] for champ in t1_champions]
    t2_champions = [champ_data['data'][champ]['id'] for champ in t2_champions]
    
    additional_columns = data['game_stats'].keys()
    additional_data = [data['game_stats'][key] for key in additional_columns]
    
    columns = [f't1_champ{i+1}id' for i in range(5)] + [f't2_champ{i+1}id' for i in range(5)] + list(additional_columns)
    df = pd.DataFrame([t1_champions + t2_champions + additional_data], columns=columns)
    return df


def model(data, predict_data=None, start=False, more=None):
    X_train, X_test, y_train, y_test = data

    clf = RandomForestClassifier(n_jobs=-1)
    clf.fit(X_train, y_train)
    print("Probability to win: ", clf.score(X_test, y_test))
    
    try:
        if predict_data is not None:
            predict_data = clean_df(predict_data, start=start, more=more, is_prediction=True)

            missing_cols = list(set(X_train.columns) - set(predict_data.columns))
            missing_data = pd.DataFrame(0, index=predict_data.index, columns=missing_cols)
            predict_data = pd.concat([predict_data, missing_data], axis=1)

            predict_data = predict_data[X_train.columns]

            prediction = clf.predict(predict_data)
            probabilities = clf.predict_proba(predict_data)
            print("Prediction for new game: ", prediction[0])
            print(f"Probability of team 1 winning: {probabilities[0][0]:.2f}")
            print(f"Probability of team 2 winning: {probabilities[0][1]:.2f}")
    except ValueError as e:
        logger.error(f"ValueError occurred while predicting new game: {str(e)}")
    return


def champ_winrate(df, champ_name):
    wins1 = len(df[(df[f't1_{champ_name}'] == 1) & (df['winner'] == 1)])
    wins2 = len(df[(df[f't2_{champ_name}'] == 1) & (df['winner'] == 2)])

    losses1 = len(df[(df[f't1_{champ_name}'] == 1) & (df['winner'] == 2)])
    losses2 = len(df[(df[f't2_{champ_name}'] == 1) & (df['winner'] == 1)])
    
    try:
        winrate = (wins1 + wins2) / (wins1 + wins2 + losses1 + losses2)
    except ZeroDivisionError:
        logger.error("ZeroDivisionError occurred while calculating winrate")
        return None
    
    return winrate


def main():
    args = args_parser()
    start = args.start
    more = args.more
    
    df = pd.read_csv('../data/game_emerald.csv')
    champ_data = json.load(open("../data/champion_info_3.json"))
    df = elab_champs(df, champ_data)

    if args.file:
        predict_data = extract_data(args.file, champ_data)
        predict_data = elab_champs(predict_data, champ_data)
    else:
        predict_data = None
        
    df = clean_df(df, start, more)
    
    X, y = df.drop('winner', axis=1), df['winner']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    data = (X_train, X_test, y_train, y_test)
    model(data, predict_data, start, more)
    
    if args.champion is not None:
        champ_name = args.champion
        winrate = champ_winrate(df, champ_name)
        if winrate is not None:
            print(f"Winrate of {champ_name}: {str(winrate)}")
        else:
            print("Error occurred")
            logger.error("Error occurred while calculating winrate")


def args_parser():
    parser = argparse.ArgumentParser(description='Predict the winner of a League of Legends game')
    parser.add_argument('-s', '--start', default=False, action='store_true', help='Start of the game, only uses starting champions')
    parser.add_argument('-m', '--more', type=int, help='More arguments, uses more data (1-5), insert number of arguments')
    parser.add_argument('-f', '--file', type=str, default="../data/predict.yaml", help='File to use for prediction')
    parser.add_argument('-c', '--champion', type=str, help='Champion name to get winrate')
    return parser.parse_args()


if __name__ == "__main__":
    main()
