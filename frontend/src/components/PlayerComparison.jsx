import React, { useState, useEffect } from 'react';
import { Search, TrendingUp, TrendingDown, User, Target, BarChart3, Award } from 'lucide-react';
import Plot from 'react-plotly.js';

const PlayerComparison = () => {
  const [player1, setPlayer1] = useState('');
  const [player2, setPlayer2] = useState('');
  const [comparisonData, setComparisonData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [players, setPlayers] = useState([]);

  useEffect(() => {
    fetchPlayers();
  }, []);

  const fetchPlayers = async () => {
    try {
      const response = await fetch('http://localhost:8000/players');
      const data = await response.json();
      setPlayers(data.players || []);
    } catch (error) {
      console.error('Error fetching players:', error);
    }
  };

  const handleComparisonSearch = async () => {
    if (!player1 || !player2) {
      setError('Please select both players');
      return;
    }

    if (player1 === player2) {
      setError('Please select different players');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/compare/players', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          player1: player1,
          player2: player2,
          season: '2023-24'
        }),
      });

      const data = await response.json();
      setComparisonData(data.comparison);
    } catch (error) {
      console.error('Error fetching comparison data:', error);
      setError('Failed to fetch player comparison data');
    } finally {
      setIsLoading(false);
    }
  };

  const renderPlayerStats = (playerData, playerName) => {
    if (!playerData.stats) return null;

    const stats = playerData.stats;

    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
          <User className="w-5 h-5 mr-2" />
          {playerName}
        </h3>

        {/* Key Stats */}
        <div className="grid grid-cols-2 gap-4 mb-6">
          <div className="text-center p-3 bg-blue-50 rounded-lg">
            <div className="text-2xl font-bold text-blue-600">{stats.points?.toFixed(1) || 'N/A'}</div>
            <div className="text-sm text-gray-600">Points/Game</div>
          </div>
          <div className="text-center p-3 bg-green-50 rounded-lg">
            <div className="text-2xl font-bold text-green-600">{stats.rebounds?.toFixed(1) || 'N/A'}</div>
            <div className="text-sm text-gray-600">Rebounds/Game</div>
          </div>
          <div className="text-center p-3 bg-purple-50 rounded-lg">
            <div className="text-2xl font-bold text-purple-600">{stats.assists?.toFixed(1) || 'N/A'}</div>
            <div className="text-sm text-gray-600">Assists/Game</div>
          </div>
          <div className="text-center p-3 bg-orange-50 rounded-lg">
            <div className="text-2xl font-bold text-orange-600">{stats.minutes?.toFixed(1) || 'N/A'}</div>
            <div className="text-sm text-gray-600">Minutes/Game</div>
          </div>
        </div>

        {/* Shooting Stats */}
        <div className="mb-6">
          <h4 className="font-semibold text-gray-700 mb-3">Shooting Efficiency</h4>
          <div className="grid grid-cols-3 gap-4 text-sm">
            <div className="text-center">
              <div className="text-lg font-bold text-gray-800">
                {stats.field_goal_pct ? (stats.field_goal_pct * 100).toFixed(1) : 'N/A'}%
              </div>
              <div className="text-xs text-gray-600">Field Goal %</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-gray-800">
                {stats.three_point_pct ? (stats.three_point_pct * 100).toFixed(1) : 'N/A'}%
              </div>
              <div className="text-xs text-gray-600">3-Point %</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-gray-800">
                {stats.free_throw_pct ? (stats.free_throw_pct * 100).toFixed(1) : 'N/A'}%
              </div>
              <div className="text-xs text-gray-600">Free Throw %</div>
            </div>
          </div>
        </div>

        {/* Defensive Stats */}
        <div>
          <h4 className="font-semibold text-gray-700 mb-3">Defensive Stats</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="flex justify-between">
              <span>Steals/Game:</span>
              <span className="font-medium">{stats.steals?.toFixed(1) || 'N/A'}</span>
            </div>
            <div className="flex justify-between">
              <span>Blocks/Game:</span>
              <span className="font-medium">{stats.blocks?.toFixed(1) || 'N/A'}</span>
            </div>
          </div>
        </div>

        {/* Games Played */}
        <div className="mt-4 pt-4 border-t">
          <div className="text-center">
            <div className="text-lg font-bold text-gray-800">{playerData.games_played || 0}</div>
            <div className="text-sm text-gray-600">Games Played</div>
          </div>
        </div>
      </div>
    );
  };

  const renderComparisonChart = () => {
    if (!comparisonData) return null;

    const player1Stats = comparisonData.player1.stats;
    const player2Stats = comparisonData.player2.stats;

    const chartData = [
      {
        x: ['Points', 'Rebounds', 'Assists', 'Steals', 'Blocks'],
        y: [
          player1Stats.points || 0,
          player1Stats.rebounds || 0,
          player1Stats.assists || 0,
          player1Stats.steals || 0,
          player1Stats.blocks || 0
        ],
        type: 'bar',
        name: player1,
        marker: { color: '#3B82F6' }
      },
      {
        x: ['Points', 'Rebounds', 'Assists', 'Steals', 'Blocks'],
        y: [
          player2Stats.points || 0,
          player2Stats.rebounds || 0,
          player2Stats.assists || 0,
          player2Stats.steals || 0,
          player2Stats.blocks || 0
        ],
        type: 'bar',
        name: player2,
        marker: { color: '#8B5CF6' }
      }
    ];

    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
          <BarChart3 className="w-5 h-5 mr-2" />
          Statistical Comparison
        </h3>
        <Plot
          data={chartData}
          layout={{
            title: `${player1} vs ${player2} - Per Game Stats`,
            xaxis: { title: 'Statistics' },
            yaxis: { title: 'Per Game Average' },
            barmode: 'group',
            height: 400,
            margin: { t: 50, b: 50, l: 50, r: 50 }
          }}
          config={{ displayModeBar: false }}
        />
      </div>
    );
  };

  const renderShootingComparison = () => {
    if (!comparisonData) return null;

    const player1Stats = comparisonData.player1.stats;
    const player2Stats = comparisonData.player2.stats;

    const shootingData = [
      {
        x: ['FG%', '3PT%', 'FT%'],
        y: [
          (player1Stats.field_goal_pct || 0) * 100,
          (player1Stats.three_point_pct || 0) * 100,
          (player1Stats.free_throw_pct || 0) * 100
        ],
        type: 'bar',
        name: player1,
        marker: { color: '#3B82F6' }
      },
      {
        x: ['FG%', '3PT%', 'FT%'],
        y: [
          (player2Stats.field_goal_pct || 0) * 100,
          (player2Stats.three_point_pct || 0) * 100,
          (player2Stats.free_throw_pct || 0) * 100
        ],
        type: 'bar',
        name: player2,
        marker: { color: '#8B5CF6' }
      }
    ];

    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
          <Target className="w-5 h-5 mr-2" />
          Shooting Efficiency Comparison
        </h3>
        <Plot
          data={shootingData}
          layout={{
            title: `${player1} vs ${player2} - Shooting Percentages`,
            xaxis: { title: 'Shooting Type' },
            yaxis: { title: 'Percentage (%)' },
            barmode: 'group',
            height: 400,
            margin: { t: 50, b: 50, l: 50, r: 50 }
          }}
          config={{ displayModeBar: false }}
        />
      </div>
    );
  };

  const renderAdvantageAnalysis = () => {
    if (!comparisonData) return null;

    const player1Stats = comparisonData.player1.stats;
    const player2Stats = comparisonData.player2.stats;

    const advantages = [];

    // Compare key stats
    if (player1Stats.points > player2Stats.points) {
      advantages.push({ player: player1, stat: 'Points', advantage: (player1Stats.points - player2Stats.points).toFixed(1) });
    } else if (player2Stats.points > player1Stats.points) {
      advantages.push({ player: player2, stat: 'Points', advantage: (player2Stats.points - player1Stats.points).toFixed(1) });
    }

    if (player1Stats.rebounds > player2Stats.rebounds) {
      advantages.push({ player: player1, stat: 'Rebounds', advantage: (player1Stats.rebounds - player2Stats.rebounds).toFixed(1) });
    } else if (player2Stats.rebounds > player1Stats.rebounds) {
      advantages.push({ player: player2, stat: 'Rebounds', advantage: (player2Stats.rebounds - player1Stats.rebounds).toFixed(1) });
    }

    if (player1Stats.assists > player2Stats.assists) {
      advantages.push({ player: player1, stat: 'Assists', advantage: (player1Stats.assists - player2Stats.assists).toFixed(1) });
    } else if (player2Stats.assists > player1Stats.assists) {
      advantages.push({ player: player2, stat: 'Assists', advantage: (player2Stats.assists - player1Stats.assists).toFixed(1) });
    }

    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
          <Award className="w-5 h-5 mr-2" />
          Statistical Advantages
        </h3>
        
        {advantages.length > 0 ? (
          <div className="space-y-3">
            {advantages.map((advantage, index) => (
              <div key={index} className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
                <div className="flex items-center">
                  <TrendingUp className="w-4 h-4 text-green-500 mr-2" />
                  <span className="font-medium text-gray-800">{advantage.player}</span>
                </div>
                <div className="text-right">
                  <div className="text-sm text-gray-600">{advantage.stat}</div>
                  <div className="font-bold text-green-600">+{advantage.advantage}</div>
                </div>
              </div>
            ))}
          </div>
        ) : (
          <p className="text-gray-600 text-center py-4">No significant statistical advantages found</p>
        )}
      </div>
    );
  };

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">Player Comparison</h1>
        <p className="text-gray-600">Compare player statistics and performance</p>
      </div>

      {/* Search Form */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Player 1</label>
            <input
              type="text"
              value={player1}
              onChange={(e) => setPlayer1(e.target.value)}
              placeholder="Enter player name"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Player 2</label>
            <input
              type="text"
              value={player2}
              onChange={(e) => setPlayer2(e.target.value)}
              placeholder="Enter player name"
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            />
          </div>

          <div className="flex items-end">
            <button
              onClick={handleComparisonSearch}
              disabled={isLoading || !player1 || !player2}
              className="w-full px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
            >
              {isLoading ? (
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
              ) : (
                <>
                  <Search className="w-4 h-4 mr-2" />
                  Compare Players
                </>
              )}
            </button>
          </div>
        </div>

        {error && (
          <div className="mt-4 p-3 bg-red-100 border border-red-400 text-red-700 rounded">
            {error}
          </div>
        )}
      </div>

      {/* Results */}
      {comparisonData && (
        <div className="space-y-8">
          {/* Player Stats Comparison */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {renderPlayerStats(comparisonData.player1, player1)}
            {renderPlayerStats(comparisonData.player2, player2)}
          </div>

          {/* Statistical Comparison Chart */}
          {renderComparisonChart()}

          {/* Shooting Comparison */}
          {renderShootingComparison()}

          {/* Advantage Analysis */}
          {renderAdvantageAnalysis()}
        </div>
      )}

      {/* Empty State */}
      {!comparisonData && !isLoading && (
        <div className="text-center py-12">
          <User className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-600 mb-2">Select Players to Compare</h3>
          <p className="text-gray-500">Enter two player names to see their statistical comparison</p>
        </div>
      )}
    </div>
  );
};

export default PlayerComparison;
