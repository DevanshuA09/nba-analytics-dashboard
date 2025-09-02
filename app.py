"""
NBA Analytics Dashboard - FastAPI Backend
Professional portfolio-ready application with predictive models
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Dict, Any
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging

# Import our custom modules
from services.features import FeatureExtractor
from services.nba_data import NBADataService
from services.llm_agent import get_nba_agent
from services.cache import get_cache_service
from services.nba_tools import NBATools

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="NBA Analytics Dashboard API",
    description="Predictive NBA analytics with team win probability and player points forecasting",
    version="1.0.0"
)

# Add CORS middleware for React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
feature_extractor = FeatureExtractor()
nba_data_service = NBADataService()
nba_agent = get_nba_agent()
cache_service = get_cache_service()
nba_tools = NBATools(cache_service=cache_service)

# Pydantic models
class TeamWinPredictionRequest(BaseModel):
    home_team: str
    away_team: str
    home_rest_days: int = 1
    away_rest_days: int = 1
    season: str = "2023-24"

class PlayerPointsPredictionRequest(BaseModel):
    player_name: str
    opponent_team: str
    is_home: bool = True
    season: str = "2023-24"

class PredictionResponse(BaseModel):
    prediction: float
    confidence: Optional[float] = None
    model_info: Dict[str, Any]
    timestamp: str

class TeamWinResponse(PredictionResponse):
    home_team: str
    away_team: str
    win_probability: float

class PlayerPointsResponse(PredictionResponse):
    player_name: str
    opponent_team: str
    predicted_points: float
    confidence_interval: Dict[str, float]

class ChatRequest(BaseModel):
    query: str
    context: Optional[Dict[str, Any]] = None

class ChatResponse(BaseModel):
    type: str
    message: str
    data: Optional[Dict[str, Any]] = None
    visualization: Optional[Dict[str, Any]] = None
    timestamp: str

class PlayerComparisonRequest(BaseModel):
    player1: str
    player2: str
    season: str = "2023-24"

class TeamMatchupRequest(BaseModel):
    team1: str
    team2: str
    season: str = "2023-24"

# Load models
def load_models():
    """Load trained models from disk"""
    models = {}
    try:
        if os.path.exists("models/team_win.pkl"):
            models['team_win'] = joblib.load("models/team_win.pkl")
            logger.info("Team win model loaded successfully")
        else:
            logger.warning("Team win model not found. Please train the model first.")
            
        if os.path.exists("models/player_points.pkl"):
            models['player_points'] = joblib.load("models/player_points.pkl")
            logger.info("Player points model loaded successfully")
        else:
            logger.warning("Player points model not found. Please train the model first.")
            
    except Exception as e:
        logger.error(f"Error loading models: {e}")
    
    return models

# Global models storage
models = load_models()

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global models
    models = load_models()
    logger.info("NBA Analytics API started successfully")

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "NBA Analytics Dashboard API",
        "version": "1.0.0",
        "endpoints": {
            "team_win_prediction": "/predict/team-win",
            "player_points_prediction": "/predict/player-points",
            "health_check": "/health",
            "docs": "/docs"
        },
        "models_loaded": {
            "team_win": "team_win" in models,
            "player_points": "player_points" in models
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": {
            "team_win": "team_win" in models,
            "player_points": "player_points" in models
        }
    }

@app.post("/predict/team-win", response_model=TeamWinResponse)
async def predict_team_win(request: TeamWinPredictionRequest):
    """Predict team win probability for a matchup"""
    try:
        if "team_win" not in models:
            raise HTTPException(status_code=503, detail="Team win model not available. Please train the model first.")
        
        # Extract features for the matchup
        features = feature_extractor.extract_team_features(
            home_team=request.home_team,
            away_team=request.away_team,
            home_rest_days=request.home_rest_days,
            away_rest_days=request.away_rest_days,
            season=request.season
        )
        
        if features is None:
            raise HTTPException(status_code=400, detail="Could not extract features for the specified teams")
        
        # Make prediction
        model_data = models["team_win"]
        model = model_data['model']
        scaler = model_data['scaler']
        
        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))
        win_probability = model.predict_proba(features_scaled)[0][1]
        
        # Calculate confidence
        if hasattr(model, 'decision_function'):
            decision_score = model.decision_function(features_scaled)[0]
            confidence = min(0.95, max(0.5, abs(decision_score) / 2 + 0.5))
        else:
            confidence = 0.75
        
        return TeamWinResponse(
            home_team=request.home_team,
            away_team=request.away_team,
            win_probability=round(win_probability, 3),
            prediction=round(win_probability, 3),
            confidence=round(confidence, 3),
            model_info={
                "model_type": type(model).__name__,
                "features_used": len(features),
                "training_date": "2024-01-01"
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in team win prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/player-points", response_model=PlayerPointsResponse)
async def predict_player_points(request: PlayerPointsPredictionRequest):
    """Predict player points for next game"""
    try:
        if "player_points" not in models:
            raise HTTPException(status_code=503, detail="Player points model not available. Please train the model first.")
        
        # Extract features for the player
        features = feature_extractor.extract_player_features(
            player_name=request.player_name,
            opponent_team=request.opponent_team,
            is_home=request.is_home,
            season=request.season
        )
        
        if features is None:
            raise HTTPException(status_code=400, detail="Could not extract features for the specified player")
        
        # Make prediction
        model_data = models["player_points"]
        model = model_data['model']
        scaler = model_data['scaler']
        
        # Scale features
        features_scaled = scaler.transform(features.reshape(1, -1))
        predicted_points = model.predict(features_scaled)[0]
        
        # Calculate confidence interval
        std_dev = 3.0
        confidence_interval = {
            "lower": round(max(0, predicted_points - 1.96 * std_dev), 1),
            "upper": round(predicted_points + 1.96 * std_dev, 1)
        }
        
        return PlayerPointsResponse(
            player_name=request.player_name,
            opponent_team=request.opponent_team,
            predicted_points=round(predicted_points, 1),
            prediction=round(predicted_points, 1),
            confidence_interval=confidence_interval,
            confidence=0.8,
            model_info={
                "model_type": type(model).__name__,
                "features_used": len(features),
                "training_date": "2024-01-01"
            },
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in player points prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/teams")
async def get_teams():
    """Get list of NBA teams"""
    try:
        teams = nba_data_service.get_team_list()
        return {"teams": teams}
    except Exception as e:
        logger.error(f"Error fetching teams: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch team list")

@app.get("/players")
async def get_players(team: Optional[str] = None):
    """Get list of NBA players, optionally filtered by team"""
    try:
        players = nba_data_service.get_player_list(team=team)
        return {"players": players}
    except Exception as e:
        logger.error(f"Error fetching players: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch player list")

@app.get("/games/today")
async def get_todays_games():
    """Get today's NBA games with predictions"""
    try:
        games = nba_data_service.get_todays_games()
        
        # Add predictions if model is available
        if "team_win" in models:
            for game in games:
                try:
                    features = feature_extractor.extract_team_features(
                        home_team=game["home_team"],
                        away_team=game["away_team"],
                        home_rest_days=game.get("home_rest_days", 1),
                        away_rest_days=game.get("away_rest_days", 1)
                    )
                    
                    if features is not None:
                        model_data = models["team_win"]
                        model = model_data['model']
                        scaler = model_data['scaler']
                        features_scaled = scaler.transform(features.reshape(1, -1))
                        win_prob = model.predict_proba(features_scaled)[0][1]
                        game["home_win_probability"] = round(win_prob, 3)
                        game["away_win_probability"] = round(1 - win_prob, 3)
                    else:
                        game["home_win_probability"] = 0.5
                        game["away_win_probability"] = 0.5
                        
                except Exception as e:
                    logger.warning(f"Could not predict for game {game['home_team']} vs {game['away_team']}: {e}")
                    game["home_win_probability"] = 0.5
                    game["away_win_probability"] = 0.5
        else:
            for game in games:
                game["home_win_probability"] = 0.5
                game["away_win_probability"] = 0.5
        
        return {"games": games, "date": datetime.now().strftime("%Y-%m-%d")}
        
    except Exception as e:
        logger.error(f"Error fetching today's games: {e}")
        raise HTTPException(status_code=500, detail="Could not fetch today's games")

@app.post("/chat/ask", response_model=ChatResponse)
async def chat_ask(request: ChatRequest):
    """AI chatbot endpoint for natural language NBA queries"""
    try:
        # Process query with NBA agent
        response = nba_agent.process_query(request.query, request.context)
        
        return ChatResponse(
            type=response["type"],
            message=response["message"],
            data=response.get("data"),
            visualization=response.get("visualization"),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error in chat query: {e}")
        raise HTTPException(status_code=500, detail=f"Chat query failed: {str(e)}")

@app.post("/compare/players")
async def compare_players(request: PlayerComparisonRequest):
    """Compare two players' statistics and performance"""
    try:
        # Get player data for both players
        player1_data = nba_tools.get_player_gamelog(request.player1, request.season, last_n_games=20)
        player2_data = nba_tools.get_player_gamelog(request.player2, request.season, last_n_games=20)
        
        if "error" in player1_data or "error" in player2_data:
            raise HTTPException(status_code=400, detail="Could not fetch player data for comparison")
        
        # Create comparison data
        comparison = {
            "player1": {
                "name": request.player1,
                "stats": player1_data["recent_averages"],
                "games_played": player1_data["games_played"]
            },
            "player2": {
                "name": request.player2,
                "stats": player2_data["recent_averages"],
                "games_played": player2_data["games_played"]
            },
            "season": request.season
        }
        
        return {
            "comparison": comparison,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error comparing players: {e}")
        raise HTTPException(status_code=500, detail=f"Player comparison failed: {str(e)}")

@app.post("/matchup")
async def team_matchup(request: TeamMatchupRequest):
    """Get head-to-head matchup data between two teams"""
    try:
        # Get head-to-head data
        h2h_data = nba_tools.get_head_to_head(request.team1, request.team2, request.season)
        
        if "error" in h2h_data:
            raise HTTPException(status_code=400, detail=h2h_data["error"])
        
        # Get individual team stats
        team1_stats = nba_tools.get_team_stats(request.team1, request.season)
        team2_stats = nba_tools.get_team_stats(request.team2, request.season)
        
        # Get recent performance
        team1_recent = nba_tools.get_team_gamelog(request.team1, request.season, last_n_games=10)
        team2_recent = nba_tools.get_team_gamelog(request.team2, request.season, last_n_games=10)
        
        matchup_data = {
            "head_to_head": h2h_data,
            "team1": {
                "name": request.team1,
                "stats": team1_stats,
                "recent_performance": team1_recent
            },
            "team2": {
                "name": request.team2,
                "stats": team2_stats,
                "recent_performance": team2_recent
            },
            "season": request.season
        }
        
        return {
            "matchup": matchup_data,
            "timestamp": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error getting team matchup: {e}")
        raise HTTPException(status_code=500, detail=f"Team matchup failed: {str(e)}")

@app.get("/cache/stats")
async def get_cache_stats():
    """Get cache statistics and health"""
    try:
        stats = cache_service.get_cache_stats()
        return {
            "cache_stats": stats,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting cache stats: {e}")
        raise HTTPException(status_code=500, detail=f"Cache stats failed: {str(e)}")

@app.delete("/cache/clear")
async def clear_cache():
    """Clear all cache data"""
    try:
        # Clear different cache patterns
        nba_cleared = cache_service.clear_pattern("nba_api_*")
        model_cleared = cache_service.clear_pattern("model_*")
        chat_cleared = cache_service.clear_pattern("chat_*")
        
        return {
            "message": "Cache cleared successfully",
            "cleared_keys": {
                "nba_api": nba_cleared,
                "model": model_cleared,
                "chat": chat_cleared
            },
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {e}")
        raise HTTPException(status_code=500, detail=f"Cache clear failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
