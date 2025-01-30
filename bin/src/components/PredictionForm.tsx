import React, { useState, useEffect } from 'react';
import Select from 'react-select';
import { PredictionRequest, GameStats } from '../types';
import { predict, getChampionWinrate, getRandomPrediction, getChampionNames } from '../api';
import { Sword, Shield } from 'lucide-react';

interface PredictionFormProps {
  isDarkMode: boolean;
}

const initialGameStats: GameStats = {
  firstBlood: 0,
  firstDragon: 0,
  firstTower: 0,
  firstInhibitor: 0,
  firstBaron: 0,
  firstRiftHerald: 0,
};

export const PredictionForm: React.FC<PredictionFormProps> = ({ isDarkMode }) => {
  const [team1Champions, setTeam1Champions] = useState(['', '', '', '', '']);
  const [team2Champions, setTeam2Champions] = useState(['', '', '', '', '']);
  const [gameStats, setGameStats] = useState<GameStats>(initialGameStats);
  const [prediction, setPrediction] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [championName, setChampionName] = useState('');
  const [winrate, setWinrate] = useState<string | null>(null);
  const [totalGames, setTotalGames] = useState<number | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [championNames, setChampionNames] = useState<string[]>([]);

  useEffect(() => {
    const fetchChampionNames = async () => {
      try {
        const names = await getChampionNames();
        setChampionNames(names);
      } catch (error) {
        console.error('Failed to fetch champion names:', error);
      }
    };

    fetchChampionNames();
  }, []);

  const handlePredict = async () => {
    setLoading(true);
    setError(null); // Clear previous errors
    try {
      const data: PredictionRequest = {
        t1_team: {
          t1_champ1: team1Champions[0],
          t1_champ2: team1Champions[1],
          t1_champ3: team1Champions[2],
          t1_champ4: team1Champions[3],
          t1_champ5: team1Champions[4],
        },
        t2_team: {
          t2_champ1: team2Champions[0],
          t2_champ2: team2Champions[1],
          t2_champ3: team2Champions[2],
          t2_champ4: team2Champions[3],
          t2_champ5: team2Champions[4],
        },
        game_stats: gameStats,
      };
  
      const result = await predict(data);
      setPrediction(result.prediction);
    } catch (error) {
      console.error('Prediction failed:', error);
      setError('Failed to make prediction. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  {error && (
    <div className="mt-4 p-4 bg-red-100 text-red-700 rounded-lg">
      {error}
    </div>
  )}

  const handleGetWinrate = async () => {
    if (!championName) return;
    try {
      const result = await getChampionWinrate(championName);
      setWinrate((result.winrate * 100).toFixed(2));
      setTotalGames(result.total_games);
    } catch (error) {
      console.error('Failed to fetch winrate:', error);
    }
  };

  const handleRandomPrediction = async () => {
    setLoading(true);
    try {
      const result = await getRandomPrediction();
      console.log('Random Prediction Result:', result);

      if (!result.teams || !result.prediction) {
        throw new Error('Invalid random prediction response');
      }

      setTeam1Champions(result.teams.team1);
      setTeam2Champions(result.teams.team2);

      setGameStats({
        firstBlood: 0,
        firstDragon: 0,
        firstTower: 0,
        firstInhibitor: 0,
        firstBaron: 0,
        firstRiftHerald: 0,
      });

      const predictedWinner = result.prediction.predicted_winner === 1 ? 'Team 1' : 'Team 2';
      setPrediction(`Predicted Winner: ${predictedWinner} (Team 1: ${(result.prediction.team1_win_probability * 100).toFixed(2)}%, Team 2: ${(result.prediction.team2_win_probability * 100).toFixed(2)}%)`);
    } catch (error) {
      console.error('Failed to fetch random prediction:', error);
      alert('Failed to generate random teams. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const updateTeamChampion = (team: number, index: number, value: string) => {
    if (team === 1) {
      const newTeam = [...team1Champions];
      newTeam[index] = value;
      setTeam1Champions(newTeam);
    } else {
      const newTeam = [...team2Champions];
      newTeam[index] = value;
      setTeam2Champions(newTeam);
    }
  };

  const inputClasses = `mb-2 w-full p-2 border rounded focus:ring-2 focus:ring-purple-500 ${
    isDarkMode
      ? 'bg-gray-800 border-gray-700 text-white placeholder-gray-400'
      : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500'
  }`;

  const selectClasses = `w-full p-2 border rounded focus:ring-2 focus:ring-purple-500 ${
    isDarkMode
      ? 'bg-gray-800 border-gray-700 text-white'
      : 'bg-white border-gray-300 text-gray-900'
  }`;

  const customStyles = {
    control: (provided: any, state: any) => ({
      ...provided,
      backgroundColor: isDarkMode ? '#1F2937' : 'white',
      borderColor: isDarkMode ? '#374151' : '#D1D5DB',
      color: isDarkMode ? 'white' : 'black',
    }),
    option: (provided: any, state: any) => ({
      ...provided,
      backgroundColor: isDarkMode ? (state.isFocused ? '#4B5563' : '#1F2937') : state.isFocused ? '#E5E7EB' : 'white',
      color: isDarkMode ? 'white' : 'black',
    }),
    singleValue: (provided: any, state: any) => ({
      ...provided,
      color: isDarkMode ? 'white' : 'black',
    }),
    input: (provided: any, state: any) => ({
      ...provided,
      color: isDarkMode ? 'white' : 'black',
    }),
  };

  return (
    <div className={`max-w-4xl mx-auto p-6 rounded-lg shadow-xl ${isDarkMode ? 'bg-gray-800' : 'bg-white'}`}>
      <h2 className="text-2xl font-bold mb-6 text-center">Game Prediction</h2>
      
      <div className="grid grid-cols-1 md:grid-cols-2 gap-8 mb-8">
        <div>
          <h3 className="text-xl font-semibold mb-4 flex items-center">
            <Sword className="mr-2 text-blue-500" />
            Team 1
          </h3>
          {team1Champions.map((champ, index) => (
            <Select
              key={`team1-${index}`}
              options={championNames.map(name => ({ value: name, label: name }))}
              value={champ ? { value: champ, label: champ } : null}
              onChange={(selectedOption) => updateTeamChampion(1, index, selectedOption?.value || '')}
              placeholder={`Champion ${index + 1}`}
              styles={customStyles}
            />
          ))}
        </div>
        
        <div>
          <h3 className="text-xl font-semibold mb-4 flex items-center">
            <Shield className="mr-2 text-red-500" />
            Team 2
          </h3>
          {team2Champions.map((champ, index) => (
            <Select
              key={`team2-${index}`}
              options={championNames.map(name => ({ value: name, label: name }))}
              value={champ ? { value: champ, label: champ } : null}
              onChange={(selectedOption) => updateTeamChampion(2, index, selectedOption?.value || '')}
              placeholder={`Champion ${index + 1}`}
              styles={customStyles}
            />
          ))}
        </div>
      </div>

      <div className="mb-8">
        <h3 className="text-xl font-semibold mb-4">Game Statistics</h3>
        <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
          {Object.entries(gameStats).map(([key, value]) => (
            <div key={key}>
              <label className={`block text-sm font-medium mb-1 ${isDarkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                {key.replace(/([A-Z])/g, ' $1').toLowerCase()}
              </label>
              <select
                value={value}
                onChange={(e) => setGameStats({ ...gameStats, [key]: Number(e.target.value) })}
                className={selectClasses}
              >
                <option value={0}>No</option>
                <option value={1}>Team1</option>
                <option value={2}>Team2</option>
              </select>
            </div>
          ))}
        </div>
      </div>

      <button
        onClick={handleRandomPrediction}
        disabled={loading}
        className="w-full mt-4 py-3 px-6 bg-green-500 text-white rounded-lg hover:bg-green-600"
      >
        {loading ? 'Generating...' : 'Generate Random Teams & Predict'}
      </button>

      <button
        onClick={handlePredict}
        disabled={loading}
        className={`w-full py-3 px-6 text-white rounded-lg focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-purple-500 transition-colors ${
          isDarkMode
            ? 'bg-purple-600 hover:bg-purple-700'
            : 'bg-purple-600 hover:bg-purple-700'
        }`}
      >
        {loading ? 'Predicting...' : 'Predict Winner'}
      </button>

      {prediction && (
        <div className={`mt-6 p-4 rounded-lg ${isDarkMode ? 'bg-gray-700' : 'bg-gray-50'}`}>
          <h4 className="text-lg font-semibold mb-2">Prediction Result:</h4>
            <p className={`text-lg text-center ${isDarkMode ? 'text-gray-400' : 'text-gray-600'}`}>
            Note: Some predictions may look the same due to the random nature of the prediction.
            </p>
          <p className={`text-xl text-center font-bold ${isDarkMode ? 'text-purple-400' : 'text-purple-600'}`}>
            {prediction}
          </p>
        </div>
      )}

      <div className="mb-6">
        <h3 className="text-xl font-semibold mb-2">Check Champion Winrate</h3>
        <input
          type="text"
          value={championName}
          onChange={(e) => setChampionName(e.target.value)}
          placeholder="Enter champion name"
          className={`mb-2 w-full p-2 border rounded focus:ring-2 focus:ring-purple-500 ${
            isDarkMode
              ? 'bg-gray-800 border-gray-700 text-white placeholder-gray-400'
              : 'bg-white border-gray-300 text-gray-900 placeholder-gray-500'
          }`}
        />
        <button
          onClick={handleGetWinrate}
          className="w-full py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600"
        >
          Get Winrate
        </button>
        {winrate && totalGames !== null && (
          <p className="mt-2 text-center font-bold text-green-500">
            Winrate: {winrate}% (Total Games: {totalGames})
          </p>
        )}
      </div>
    </div>
  );
};