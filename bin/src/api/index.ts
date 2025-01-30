import { PredictionRequest } from '../types';

const API_URL = 'http://127.0.0.1:8008';

export const login = async (username: string, password: string) => {
  const response = await fetch(`${API_URL}/login`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ username, password }),
  });
  return response.json();
};

export const register = async (username: string, password: string) => {
  const response = await fetch(`${API_URL}/register`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ username, password }),
  });
  return response.json();
};

export const predict = async (data: PredictionRequest) => {
  try {
    const response = await fetch(`${API_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(data),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    return response.json();
  } catch (error) {
    console.error('Error during prediction:', error);
    throw error;
  }
};

export const getChampionNames = async (): Promise<string[]> => {
  const response = await fetch(`${API_URL}/champion_names`);
  if (!response.ok) {
    throw new Error('Failed to fetch champion names');
  }
  return response.json();
};

export const getChampionWinrate = async (championName: string) => {
  const response = await fetch(`${API_URL}/champion/${championName}/winrate`);
  return response.json();
};


export const getRandomPrediction = async () => {
  const response = await fetch(`${API_URL}/predict/random`);
  if (!response.ok) {
    throw new Error('Failed to fetch random prediction');
  }
  const data = await response.json();
  console.log('Random Prediction API Response:', data);
  return data;
};