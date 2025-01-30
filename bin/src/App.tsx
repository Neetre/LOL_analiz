import React, { useState } from 'react';
import { PredictionForm } from './components/PredictionForm';
import { Gamepad2, Sun, Moon, Info } from 'lucide-react';

function App() {
  const [isDarkMode, setIsDarkMode] = useState(false);
  const [showInfo, setShowInfo] = useState(false);

  return (
    <div className={`min-h-screen transition-colors duration-200 ${isDarkMode ? 'bg-gray-900 text-white' : 'bg-gray-100 text-gray-900'}`}>
      <div className="container mx-auto py-8 px-4">
        <header className="mb-8 text-center relative">
          <div className="absolute right-0 top-0 flex gap-2">
            <button
              onClick={() => setIsDarkMode(!isDarkMode)}
              className={`p-2 rounded-lg ${isDarkMode ? 'bg-gray-800 hover:bg-gray-700' : 'bg-white hover:bg-gray-100'} transition-colors duration-200`}
            >
              {isDarkMode ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
            <button
              onClick={() => setShowInfo(!showInfo)}
              className={`p-2 rounded-lg ${isDarkMode ? 'bg-gray-800 hover:bg-gray-700' : 'bg-white hover:bg-gray-100'} transition-colors duration-200`}
            >
              <Info className="w-5 h-5" />
            </button>
          </div>
          <div className="flex items-center justify-center mb-4">
            <Gamepad2 className={`w-12 h-12 ${isDarkMode ? 'text-purple-400' : 'text-purple-600'}`} />
          </div>
          <h1 className="text-4xl font-bold">LoL Game Predictor</h1>
          <p className={`mt-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
            Predict game outcomes based on team compositions
          </p>
        </header>

        {showInfo && (
          <div className={`mb-8 p-6 rounded-lg ${isDarkMode ? 'bg-gray-800' : 'bg-white'} shadow-xl`}>
            <h2 className="text-xl font-bold mb-4">How to Use the Predictor</h2>
            <div className="space-y-4">
              <div>
                <h3 className="font-semibold mb-2">Team Composition</h3>
                <p className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>
                  Enter the champion names for both teams exactly as they appear in the game (e.g., "Ahri", "Lee Sin", "Kai'Sa").
                </p>
              </div>
              <div>
                <h3 className="font-semibold mb-2">Game Statistics</h3>
                <p className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>
                  Select "No" if the statistic is not relevant to the prediction. Otherwise, select the team that has the advantage.
                </p>
                <ul className={`list-disc ml-6 mt-2 ${isDarkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                  <li>First Blood - First kill of the game</li>
                  <li>First Dragon - First dragon objective</li>
                  <li>First Tower - First tower destroyed</li>
                  <li>First Inhibitor - First inhibitor destroyed</li>
                  <li>First Baron - First Baron Nashor kill</li>
                  <li>First Rift Herald - First Rift Herald capture</li>
                </ul>
              </div>
              <div>
                <h3 className="font-semibold mb-2">Random Team Generation</h3>
                <p className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>
                  Click "Generate Random Teams & Predict" to quickly create random team compositions for prediction.
                  This feature is useful for testing the predictor or exploring different scenarios, expecially when you're not sure which champions to use.
                </p>
              </div>
              <div>
                <h3 className="font-semibold mb-2">Prediction Results</h3>
                <p className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>
                  After submitting, the predictor will analyze the team compositions and game statistics to predict the likely winner of the match.
                </p>
              </div>
              <div>
                <h3 className="font-semibold mb-2">Check Champion Winrate</h3>
                <p className={isDarkMode ? 'text-gray-300' : 'text-gray-600'}>
                  Enter the name of a champion to see their winrate in the current patch.
                </p>
              </div>
            </div>
          </div>
        )}

        <PredictionForm isDarkMode={isDarkMode} />
      </div>
    </div>
  );
}

export default App;