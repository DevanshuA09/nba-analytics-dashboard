import React, { useState } from 'react'
import { Target, TrendingUp, User } from 'lucide-react'

const PlayerForecastCard = ({ players, teams }) => {
  const [selectedPlayer, setSelectedPlayer] = useState('')
  const [selectedOpponent, setSelectedOpponent] = useState('')
  const [isHome, setIsHome] = useState(true)
  const [prediction, setPrediction] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState(null)

  const handlePrediction = async () => {
    if (!selectedPlayer || !selectedOpponent) {
      setError('Please select both player and opponent team')
      return
    }

    try {
      setLoading(true)
      setError(null)

      const response = await fetch('/api/predict/player-points', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          player_name: selectedPlayer,
          opponent_team: selectedOpponent,
          is_home: isHome,
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

  const getPointsColor = (points) => {
    if (points >= 25) return '#28a745'
    if (points >= 15) return '#ffc107'
    return '#dc3545'
  }

  const getPerformanceLevel = (points) => {
    if (points >= 25) return 'Excellent'
    if (points >= 20) return 'Good'
    if (points >= 15) return 'Average'
    return 'Below Average'
  }

  return (
    <div className="card">
      <h2>
        <Target size={24} />
        Player Points Forecast
      </h2>

      <div className="form-group">
        <label>Player</label>
        <select 
          value={selectedPlayer} 
          onChange={(e) => setSelectedPlayer(e.target.value)}
        >
          <option value="">Select Player</option>
          {players.map(player => (
            <option key={player.id} value={player.name}>
              {player.name} ({player.team})
            </option>
          ))}
        </select>
      </div>

      <div className="form-group">
        <label>Opponent Team</label>
        <select 
          value={selectedOpponent} 
          onChange={(e) => setSelectedOpponent(e.target.value)}
        >
          <option value="">Select Opponent</option>
          {teams.map(team => (
            <option key={team.id} value={team.name}>
              {team.name}
            </option>
          ))}
        </select>
      </div>

      <div className="form-group">
        <label>
          <input 
            type="checkbox" 
            checked={isHome}
            onChange={(e) => setIsHome(e.target.checked)}
            style={{ marginRight: '8px' }}
          />
          Playing at Home
        </label>
      </div>

      <button 
        className="btn" 
        onClick={handlePrediction}
        disabled={loading}
      >
        {loading ? 'Predicting...' : 'Get Points Forecast'}
      </button>

      {error && (
        <div className="error">
          {error}
        </div>
      )}

      {prediction && (
        <div className="prediction-result">
          <h3>
            <TrendingUp size={20} />
            Points Forecast
          </h3>
          <div className="prediction-value" style={{ color: getPointsColor(prediction.predicted_points) }}>
            {prediction.predicted_points.toFixed(1)}
          </div>
          <p><strong>{selectedPlayer}</strong> predicted points</p>
          <p className="confidence">
            Performance Level: {getPerformanceLevel(prediction.predicted_points)}
          </p>
          
          <div style={{ marginTop: '15px', padding: '15px', background: '#f8f9fa', borderRadius: '8px' }}>
            <h4 style={{ margin: '0 0 10px 0', color: '#333' }}>
              <User size={16} />
              Confidence Interval
            </h4>
            <p style={{ margin: '5px 0', color: '#666' }}>
              <strong>Range:</strong> {prediction.confidence_interval.lower} - {prediction.confidence_interval.upper} points
            </p>
            <p style={{ margin: '5px 0', color: '#666' }}>
              <strong>Opponent:</strong> {selectedOpponent}
            </p>
            <p style={{ margin: '5px 0', color: '#666' }}>
              <strong>Venue:</strong> {isHome ? 'Home' : 'Away'}
            </p>
          </div>
        </div>
      )}

      <div style={{ marginTop: '30px', padding: '20px', background: '#f8f9fa', borderRadius: '10px' }}>
        <h3 style={{ margin: '0 0 15px 0', color: '#333' }}>
          <TrendingUp size={20} />
          Model Information
        </h3>
        <p style={{ margin: '5px 0', color: '#666', fontSize: '0.9rem' }}>
          • Uses rolling 10-game player statistics
        </p>
        <p style={{ margin: '5px 0', color: '#666', fontSize: '0.9rem' }}>
          • Considers opponent defensive rating
        </p>
        <p style={{ margin: '5px 0', color: '#666', fontSize: '0.9rem' }}>
          • Accounts for home/away advantage
        </p>
        <p style={{ margin: '5px 0', color: '#666', fontSize: '0.9rem' }}>
          • Includes player usage rate and pace
        </p>
      </div>
    </div>
  )
}

export default PlayerForecastCard
