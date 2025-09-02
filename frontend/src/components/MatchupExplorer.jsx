import React, { useState, useEffect } from 'react';
import { Search, TrendingUp, TrendingDown, Users, Calendar, Target, BarChart3 } from 'lucide-react';
import Plot from 'react-plotly.js';

const MatchupExplorer = () => {
  const [team1, setTeam1] = useState('');
  const [team2, setTeam2] = useState('');
  const [matchupData, setMatchupData] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState(null);
  const [teams, setTeams] = useState([]);

  useEffect(() => {
    fetchTeams();
  }, []);

  const fetchTeams = async () => {
    try {
      const response = await fetch('http://localhost:8000/teams');
      const data = await response.json();
      setTeams(data.teams || []);
    } catch (error) {
      console.error('Error fetching teams:', error);
    }
  };

  const handleMatchupSearch = async () => {
    if (!team1 || !team2) {
      setError('Please select both teams');
      return;
    }

    if (team1 === team2) {
      setError('Please select different teams');
      return;
    }

    setIsLoading(true);
    setError(null);

    try {
      const response = await fetch('http://localhost:8000/matchup', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          team1: team1,
          team2: team2,
          season: '2023-24'
        }),
      });

      const data = await response.json();
      setMatchupData(data.matchup);
    } catch (error) {
      console.error('Error fetching matchup data:', error);
      setError('Failed to fetch matchup data');
    } finally {
      setIsLoading(false);
    }
  };

  const renderTeamStats = (teamData, teamName) => {
    if (!teamData.stats) return null;

    const stats = teamData.stats;
    const recent = teamData.recent_performance;

    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
          <Users className="w-5 h-5 mr-2" />
          {teamName}
        </h3>

        {/* Record */}
        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="text-center">
            <div className="text-2xl font-bold text-green-600">{stats.record?.wins || 0}</div>
            <div className="text-sm text-gray-600">Wins</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-red-600">{stats.record?.losses || 0}</div>
            <div className="text-sm text-gray-600">Losses</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">
              {stats.record?.win_percentage ? (stats.record.win_percentage * 100).toFixed(1) : 0}%
            </div>
            <div className="text-sm text-gray-600">Win %</div>
          </div>
        </div>

        {/* Offensive Stats */}
        <div className="mb-6">
          <h4 className="font-semibold text-gray-700 mb-3">Offensive Stats</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="flex justify-between">
              <span>Points/Game:</span>
              <span className="font-medium">{stats.offensive_stats?.points_per_game?.toFixed(1) || 'N/A'}</span>
            </div>
            <div className="flex justify-between">
              <span>FG%:</span>
              <span className="font-medium">{stats.offensive_stats?.field_goal_pct ? (stats.offensive_stats.field_goal_pct * 100).toFixed(1) : 'N/A'}%</span>
            </div>
            <div className="flex justify-between">
              <span>3PT%:</span>
              <span className="font-medium">{stats.offensive_stats?.three_point_pct ? (stats.offensive_stats.three_point_pct * 100).toFixed(1) : 'N/A'}%</span>
            </div>
            <div className="flex justify-between">
              <span>Assists/Game:</span>
              <span className="font-medium">{stats.offensive_stats?.assists_per_game?.toFixed(1) || 'N/A'}</span>
            </div>
          </div>
        </div>

        {/* Defensive Stats */}
        <div className="mb-6">
          <h4 className="font-semibold text-gray-700 mb-3">Defensive Stats</h4>
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div className="flex justify-between">
              <span>Opp. Points/Game:</span>
              <span className="font-medium">{stats.defensive_stats?.opponent_points_per_game?.toFixed(1) || 'N/A'}</span>
            </div>
            <div className="flex justify-between">
              <span>Steals/Game:</span>
              <span className="font-medium">{stats.defensive_stats?.steals_per_game?.toFixed(1) || 'N/A'}</span>
            </div>
            <div className="flex justify-between">
              <span>Blocks/Game:</span>
              <span className="font-medium">{stats.defensive_stats?.blocks_per_game?.toFixed(1) || 'N/A'}</span>
            </div>
            <div className="flex justify-between">
              <span>Opp. FG%:</span>
              <span className="font-medium">{stats.defensive_stats?.opponent_fg_pct ? (stats.defensive_stats.opponent_fg_pct * 100).toFixed(1) : 'N/A'}%</span>
            </div>
          </div>
        </div>

        {/* Recent Performance */}
        {recent && (
          <div>
            <h4 className="font-semibold text-gray-700 mb-3">Recent Performance (Last 10 Games)</h4>
            <div className="grid grid-cols-3 gap-4 text-sm">
              <div className="text-center">
                <div className="text-lg font-bold text-green-600">{recent.recent_record?.wins || 0}</div>
                <div className="text-xs text-gray-600">Recent Wins</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-red-600">{recent.recent_record?.losses || 0}</div>
                <div className="text-xs text-gray-600">Recent Losses</div>
              </div>
              <div className="text-center">
                <div className="text-lg font-bold text-blue-600">
                  {recent.recent_record?.win_percentage ? (recent.recent_record.win_percentage * 100).toFixed(1) : 0}%
                </div>
                <div className="text-xs text-gray-600">Recent Win %</div>
              </div>
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderHeadToHead = (h2hData) => {
    if (!h2hData || h2hData.games_played === 0) {
      return (
        <div className="bg-white rounded-lg shadow-md p-6">
          <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
            <Target className="w-5 h-5 mr-2" />
            Head-to-Head
          </h3>
          <p className="text-gray-600">No head-to-head games found this season.</p>
        </div>
      );
    }

    return (
      <div className="bg-white rounded-lg shadow-md p-6">
        <h3 className="text-xl font-bold text-gray-800 mb-4 flex items-center">
          <Target className="w-5 h-5 mr-2" />
          Head-to-Head
        </h3>

        <div className="grid grid-cols-3 gap-4 mb-6">
          <div className="text-center">
            <div className="text-2xl font-bold text-blue-600">{h2hData.team1_wins}</div>
            <div className="text-sm text-gray-600">{team1} Wins</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-gray-600">{h2hData.games_played}</div>
            <div className="text-sm text-gray-600">Total Games</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-purple-600">{h2hData.team2_wins}</div>
            <div className="text-sm text-gray-600">{team2} Wins</div>
          </div>
        </div>

        {h2hData.games && h2hData.games.length > 0 && (
          <div>
            <h4 className="font-semibold text-gray-700 mb-3">Recent Games</h4>
            <div className="space-y-2">
              {h2hData.games.slice(0, 5).map((game, index) => (
                <div key={index} className="flex justify-between items-center p-2 bg-gray-50 rounded">
                  <span className="text-sm text-gray-600">{new Date(game.date).toLocaleDateString()}</span>
                  <span className="text-sm font-medium">
                    {game.home_team} {game.home_score} - {game.away_score} {game.away_team}
                  </span>
                  <span className={`text-sm font-bold ${game.winner === team1 ? 'text-blue-600' : 'text-purple-600'}`}>
                    {game.winner} W
                  </span>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  };

  const renderComparisonChart = () => {
    if (!matchupData) return null;

    const team1Stats = matchupData.team1.stats;
    const team2Stats = matchupData.team2.stats;

    const chartData = [
      {
        x: ['Points/Game', 'FG%', '3PT%', 'Assists/Game', 'Rebounds/Game'],
        y: [
          team1Stats.offensive_stats?.points_per_game || 0,
          (team1Stats.offensive_stats?.field_goal_pct || 0) * 100,
          (team1Stats.offensive_stats?.three_point_pct || 0) * 100,
          team1Stats.offensive_stats?.assists_per_game || 0,
          team1Stats.offensive_stats?.rebounds_per_game || 0
        ],
        type: 'bar',
        name: team1,
        marker: { color: '#3B82F6' }
      },
      {
        x: ['Points/Game', 'FG%', '3PT%', 'Assists/Game', 'Rebounds/Game'],
        y: [
          team2Stats.offensive_stats?.points_per_game || 0,
          (team2Stats.offensive_stats?.field_goal_pct || 0) * 100,
          (team2Stats.offensive_stats?.three_point_pct || 0) * 100,
          team2Stats.offensive_stats?.assists_per_game || 0,
          team2Stats.offensive_stats?.rebounds_per_game || 0
        ],
        type: 'bar',
        name: team2,
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
            title: `${team1} vs ${team2} - Offensive Stats`,
            xaxis: { title: 'Statistics' },
            yaxis: { title: 'Values' },
            barmode: 'group',
            height: 400,
            margin: { t: 50, b: 50, l: 50, r: 50 }
          }}
          config={{ displayModeBar: false }}
        />
      </div>
    );
  };

  return (
    <div className="max-w-7xl mx-auto p-6">
      <div className="mb-8">
        <h1 className="text-3xl font-bold text-gray-800 mb-2">Team Matchup Explorer</h1>
        <p className="text-gray-600">Compare teams and analyze head-to-head matchups</p>
      </div>

      {/* Search Form */}
      <div className="bg-white rounded-lg shadow-md p-6 mb-8">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Team 1</label>
            <select
              value={team1}
              onChange={(e) => setTeam1(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">Select Team 1</option>
              {teams.map((team) => (
                <option key={team} value={team}>{team}</option>
              ))}
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Team 2</label>
            <select
              value={team2}
              onChange={(e) => setTeam2(e.target.value)}
              className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-500"
            >
              <option value="">Select Team 2</option>
              {teams.map((team) => (
                <option key={team} value={team}>{team}</option>
              ))}
            </select>
          </div>

          <div className="flex items-end">
            <button
              onClick={handleMatchupSearch}
              disabled={isLoading || !team1 || !team2}
              className="w-full px-4 py-2 bg-blue-500 text-white rounded-lg hover:bg-blue-600 disabled:opacity-50 disabled:cursor-not-allowed transition-colors flex items-center justify-center"
            >
              {isLoading ? (
                <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white"></div>
              ) : (
                <>
                  <Search className="w-4 h-4 mr-2" />
                  Analyze Matchup
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
      {matchupData && (
        <div className="space-y-8">
          {/* Head-to-Head */}
          {renderHeadToHead(matchupData.head_to_head)}

          {/* Team Stats Comparison */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
            {renderTeamStats(matchupData.team1, team1)}
            {renderTeamStats(matchupData.team2, team2)}
          </div>

          {/* Comparison Chart */}
          {renderComparisonChart()}
        </div>
      )}

      {/* Empty State */}
      {!matchupData && !isLoading && (
        <div className="text-center py-12">
          <Users className="w-16 h-16 text-gray-400 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-600 mb-2">Select Teams to Compare</h3>
          <p className="text-gray-500">Choose two teams to see their matchup analysis and statistics</p>
        </div>
      )}
    </div>
  );
};

export default MatchupExplorer;
