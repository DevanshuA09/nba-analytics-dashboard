#!/usr/bin/env python3
"""
System Test Script for NBA Analytics Dashboard
Tests the complete pipeline and API functionality
"""

import requests
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_api_endpoints():
    """Test API endpoints"""
    logger.info("Testing API endpoints...")
    
    base_url = "http://localhost:8000"
    
    # Test health endpoint
    try:
        response = requests.get(f"{base_url}/health", timeout=10)
        if response.status_code == 200:
            logger.info("✅ Health endpoint working")
        else:
            logger.error("❌ Health endpoint failed")
            return False
    except Exception as e:
        logger.error(f"❌ Health endpoint error: {e}")
        return False
    
    # Test team prediction
    try:
        team_data = {
            "home_team": "Golden State Warriors",
            "away_team": "Los Angeles Lakers",
            "home_rest_days": 1,
            "away_rest_days": 2,
            "season": "2023-24"
        }
        
        response = requests.post(
            f"{base_url}/predict/team-win",
            json=team_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ Team prediction working: {result['win_probability']:.3f}")
        else:
            logger.error(f"❌ Team prediction failed: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Team prediction error: {e}")
        return False
    
    # Test player prediction
    try:
        player_data = {
            "player_name": "LeBron James",
            "opponent_team": "Golden State Warriors",
            "is_home": True,
            "season": "2023-24"
        }
        
        response = requests.post(
            f"{base_url}/predict/player-points",
            json=player_data,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            logger.info(f"✅ Player prediction working: {result['predicted_points']:.1f} points")
        else:
            logger.error(f"❌ Player prediction failed: {response.status_code}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Player prediction error: {e}")
        return False
    
    # Test teams endpoint
    try:
        response = requests.get(f"{base_url}/teams", timeout=10)
        if response.status_code == 200:
            teams = response.json()['teams']
            logger.info(f"✅ Teams endpoint working: {len(teams)} teams")
        else:
            logger.error("❌ Teams endpoint failed")
            return False
    except Exception as e:
        logger.error(f"❌ Teams endpoint error: {e}")
        return False
    
    # Test games endpoint
    try:
        response = requests.get(f"{base_url}/games/today", timeout=10)
        if response.status_code == 200:
            games = response.json()['games']
            logger.info(f"✅ Games endpoint working: {len(games)} games")
        else:
            logger.error("❌ Games endpoint failed")
            return False
    except Exception as e:
        logger.error(f"❌ Games endpoint error: {e}")
        return False
    
    return True

def test_model_files():
    """Test if model files exist"""
    logger.info("Testing model files...")
    
    models_dir = Path("models")
    
    # Check team model
    team_model = models_dir / "team_win_final.pkl"
    if team_model.exists():
        logger.info("✅ Team model file exists")
    else:
        logger.warning("⚠️ Team model file not found (using demo model)")
    
    # Check player model
    player_model = models_dir / "player_pts_final.pkl"
    if player_model.exists():
        logger.info("✅ Player model file exists")
    else:
        logger.warning("⚠️ Player model file not found (using demo model)")
    
    return True

def test_inference_functions():
    """Test inference functions directly"""
    logger.info("Testing inference functions...")
    
    try:
        from inference import predict_team_win, predict_player_points
        
        # Test team prediction
        team_input = {
            'home_team': 'Golden State Warriors',
            'away_team': 'Los Angeles Lakers',
            'home_rest_days': 1,
            'away_rest_days': 2,
            'season': '2023-24'
        }
        
        team_result = predict_team_win(team_input)
        logger.info(f"✅ Team inference working: {team_result['win_probability']:.3f}")
        
        # Test player prediction
        player_input = {
            'player_name': 'LeBron James',
            'opponent_team': 'Golden State Warriors',
            'is_home': True,
            'season': '2023-24'
        }
        
        player_result = predict_player_points(player_input)
        logger.info(f"✅ Player inference working: {player_result['predicted_points']:.1f} points")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Inference functions error: {e}")
        return False

def main():
    """Run all tests"""
    logger.info("Starting NBA Analytics System Tests...")
    
    # Test model files
    test_model_files()
    
    # Test inference functions
    inference_ok = test_inference_functions()
    
    # Test API endpoints (requires server to be running)
    logger.info("Testing API endpoints (make sure server is running)...")
    api_ok = test_api_endpoints()
    
    # Summary
    logger.info("=" * 50)
    logger.info("TEST SUMMARY")
    logger.info("=" * 50)
    
    if inference_ok:
        logger.info("✅ Inference functions: PASSED")
    else:
        logger.error("❌ Inference functions: FAILED")
    
    if api_ok:
        logger.info("✅ API endpoints: PASSED")
    else:
        logger.error("❌ API endpoints: FAILED")
        logger.info("💡 Make sure to start the server with: python start_server.py")
    
    if inference_ok and api_ok:
        logger.info("🎉 All tests passed! System is working correctly.")
    else:
        logger.error("❌ Some tests failed. Check the logs above.")
    
    logger.info("=" * 50)

if __name__ == "__main__":
    main()
