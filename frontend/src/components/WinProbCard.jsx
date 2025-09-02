import React, { useState } from 'react'
import { BarChart3, RefreshCw, Trophy } from 'lucide-react'

const WinProbCard = ({ games, teams, onRefresh }) => {
  const [loading, setLoading] = useState(false)
  const [selectedHomeTeam, setSelectedHomeTeam] = useState('')
  const [selectedAwayTeam, setSelectedAwayTeam] = useState('')
  const [prediction, setPrediction] = useState(null)
  const [error, setError] = useState(null)

  const handlePrediction = async () => {
    if (!selectedHomeTeam || !selectedAwayTeam) {
      setError('Please select both teams')
      return
    }

    if (selectedHomeTeam === selectedAwayTeam) {
      setError('Please select different teams')
      return
    }

    try {
      setLoading(true)
      setError(null)

      const response = await fetch('/api/predict/team-win', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          home_team: selectedHomeTeam,
          away_team: selectedAwayTeam,
          home_rest_days: 1,
          away_rest_days: 1,
          season: '2023-24'
        })
      })

      if (!response.ok) {
        throw new Error('Failed to get prediction')
      }

      const data = await response.json()
      setPrediction(data)
    } catch (err) {
      console.error('Error getting prediction:', err)
      setError('Failed to get prediction. Please try again.')
    } finally {
      setLoading(false)
    }
  }

  const getWinProbabilityColor = (probability) => {
    if (probability >= 0.6) return '#28a745'
    if (probability >= 0.4) return '#ffc107'
    return '#dc3545'
  }

  const getWinProbabilityText = (probability) => {
    if (probability >= 0.6) return 'Strong Favorite'
    if (probability >= 0.4) return 'Even Match'
    return 'Underdog'
  }

  return (
    <div className="card">
      <h2>
        <BarChart3 size={24} />
        Team Win Probability
      </h2>

      <div className="form-group">
        <label>Home Team</label>
        <select 
          value={selectedHomeTeam} 
          onChange={(e) => setSelectedHomeTeam(e.target.value)}
        >
          <option value="">Select Home Team</option>
          {teams.map(team => (
            <option key={team.id} value={team.name}>
              {team.name}
            </option>
          ))}
        </select>
      </div>

      <div className="form-group">
        <label>Away Team</label>
        <select 
          value={selectedAwayTeam} 
          onChange={(e) => setSelectedAwayTeam(e.target.value)}
        >
          <option value="">Select Away Team</option>
          {teams.map(team => (
            <option key={team.id} value={team.name}>
              {team.name}
            </option>
          ))}
        </select>
      </div>

      <button 
        className="btn" 
        onClick={handlePrediction}
        disabled={loading}
      >
        {loading ? 'Predicting...' : 'Get Win Probability'}
      </button>

      {error && (
        <div className="error">
          {error}
        </div>
      )}

      {prediction && (
        <div className="prediction-result">
          <h3>
            <Trophy size={20} />
            Prediction Result
          </h3>
          <div className="prediction-value" style={{ color: getWinProbabilityColor(prediction.win_probability) }}>
            {(prediction.win_probability * 100).toFixed(1)}%
          </div>
          <p><strong>{selectedHomeTeam}</strong> win probability</p>
          <p className="confidence">
            Confidence: {(prediction.confidence * 100).toFixed(1)}% | 
            Status: {getWinProbabilityText(prediction.win_probability)}
          </p>
        </div>
      )}

      {games.length > 0 && (
        <div style={{ marginTop: '30px' }}>
          <h3>Today's Games</h3>
          <div className="games-list">
            {games.map(game => (
              <div key={game.game_id} className="game-card">
                <div className="game-header">
                  <div className="teams">
                    {game.home_team} vs {game.away_team}
                  </div>
                  <div className="game-time">
                    {game.game_time}
                  </div>
                </div>
                
                <div className="win-probability">
                  <span className="prob-text">
                    {game.away_team}<br />
                    {(game.away_win_probability * 100).toFixed(1)}%
                  </span>
                  <div className="prob-bar">
                    <div 
                      className="prob-fill" 
                      style={{ 
                        width: `${game.away_win_probability * 100}%`,
                        background: getWinProbabilityColor(game.away_win_probability)
                      }}
                    />
                  </div>
                  <span className="prob-text">
                    {game.home_team}<br />
                    {(game.home_win_probability * 100).toFixed(1)}%
                  </span>
                </div>
              </div>
            ))}
          </div>
          
          <button 
            className="btn" 
            onClick={onRefresh}
            style={{ marginTop: '15px', width: '100%' }}
          >
            <RefreshCw size={16} />
            Refresh Games
          </button>
        </div>
      )}
    </div>
  )
}

export default WinProbCard
