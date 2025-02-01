export interface Team {
  t1_champ1: string;
  t1_champ2: string;
  t1_champ3: string;
  t1_champ4: string;
  t1_champ5: string;
}

export interface Team2 {
  t2_champ1: string;
  t2_champ2: string;
  t2_champ3: string;
  t2_champ4: string;
  t2_champ5: string;
}

export interface GameStats {
  firstBlood: number;
  firstDragon: number;
  firstTower: number;
  firstInhibitor: number;
  firstBaron: number;
  firstRiftHerald: number;
}

export interface PredictionRequest {
  t1_team: Team;
  t2_team: Team2;
  game_stats: GameStats;
}