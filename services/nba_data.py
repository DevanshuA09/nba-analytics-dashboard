"""
NBA Data Service
Handles data fetching and caching for NBA API calls
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Any
from datetime import datetime, timedelta
import logging
try:
    from nba_api.stats.endpoints import (
        leaguedashteamstats, leaguedashplayerstats,
        scoreboard, teamgamelog, playergamelog
    )
    from nba_api.stats.static import teams, players
    NBA_API_AVAILABLE = True
except ImportError:
    NBA_API_AVAILABLE = False
    from .demo_data import DemoDataGenerator

logger = logging.getLogger(__name__)

class NBADataService:
    """Service for fetching and managing NBA data"""
    
    def __init__(self):
        self.cache = {}
        self.cache_timeout = 300  # 5 minutes
        if not NBA_API_AVAILABLE:
            self.demo_generator = DemoDataGenerator()
        
    def get_team_list(self) -> List[Dict[str, Any]]:
        """Get list of all NBA teams"""
        if not NBA_API_AVAILABLE:
            return self.demo_generator.get_team_list()
        
        try:
            team_dict = teams.get_teams()
            return [
                {
                    "id": team["id"],
                    "name": team["full_name"],
                    "abbreviation": team["abbreviation"],
                    "city": team["city"],
                    "conference": team["conference"],
                    "division": team["division"]
                }
                for team in team_dict
            ]
        except Exception as e:
            logger.error(f"Error fetching team list: {e}")
            return self.demo_generator.get_team_list()
    
    def get_player_list(self, team: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get list of NBA players, optionally filtered by team"""
        if not NBA_API_AVAILABLE:
            return self.demo_generator.get_player_list(team)
        
        try:
            player_dict = players.get_players()
            
            if team:
                # Filter by team (simplified - would need team_id mapping)
                team_players = [p for p in player_dict if p.get("team", {}).get("full_name") == team]
            else:
                team_players = player_dict
            
            return [
                {
                    "id": player["id"],
                    "name": player["full_name"],
                    "first_name": player["first_name"],
                    "last_name": player["last_name"],
                    "team": player.get("team", {}).get("full_name", "Free Agent")
                }
                for player in team_players
            ]
        except Exception as e:
            logger.error(f"Error fetching player list: {e}")
            return self.demo_generator.get_player_list(team)
    
    def get_todays_games(self) -> List[Dict[str, Any]]:
        """Get today's NBA games"""
        if not NBA_API_AVAILABLE:
            return self.demo_generator.get_todays_games()
        
        try:
            # Get today's date
            today = datetime.now().strftime("%Y-%m-%d")
            
            # Get scoreboard for today
            scoreboard_data = scoreboard.Scoreboard(game_date=today)
            games_df = scoreboard_data.get_data_frames()[0]
            
            if games_df.empty:
                # If no games today, return demo games
                return self.demo_generator.get_todays_games()
            
            games = []
            for _, game in games_df.iterrows():
                games.append({
                    "game_id": game["GAME_ID"],
                    "home_team": game["HOME_TEAM_NAME"],
                    "away_team": game["VISITOR_TEAM_NAME"],
                    "home_team_id": game["HOME_TEAM_ID"],
                    "away_team_id": game["VISITOR_TEAM_ID"],
                    "game_time": game["GAME_STATUS_TEXT"],
                    "home_rest_days": 1,  # Default value
                    "away_rest_days": 1   # Default value
                })
            
            return games
            
        except Exception as e:
            logger.error(f"Error fetching today's games: {e}")
            return self.demo_generator.get_todays_games()
    
    def get_team_stats(self, team_id: int, season: str = "2023-24") -> Optional[Dict[str, Any]]:
        """Get team statistics for a specific team"""
        try:
            # Get team dashboard
            dashboard = leaguedashteamstats.LeagueDashTeamStats(
                season=season,
                team_id_nullable=team_id
            )
            df = dashboard.get_data_frames()[0]
            
            if df.empty:
                return None
            
            team_stats = df.iloc[0].to_dict()
            return team_stats
            
        except Exception as e:
            logger.error(f"Error fetching team stats for team {team_id}: {e}")
            return None
    
    def get_player_stats(self, player_id: int, season: str = "2023-24") -> Optional[Dict[str, Any]]:
        """Get player statistics for a specific player"""
        try:
            # Get player dashboard
            dashboard = leaguedashplayerstats.LeagueDashPlayerStats(
                season=season,
                player_id_nullable=player_id
            )
            df = dashboard.get_data_frames()[0]
            
            if df.empty:
                return None
            
            player_stats = df.iloc[0].to_dict()
            return player_stats
            
        except Exception as e:
            logger.error(f"Error fetching player stats for player {player_id}: {e}")
            return None
    
    def get_team_game_log(self, team_id: int, season: str = "2023-24", 
                         games: int = 10) -> Optional[pd.DataFrame]:
        """Get team game log for recent games"""
        try:
            game_log = teamgamelog.TeamGameLog(team_id=team_id, season=season)
            df = game_log.get_data_frames()[0]
            
            if len(df) > games:
                df = df.head(games)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching team game log for team {team_id}: {e}")
            return None
    
    def get_player_game_log(self, player_id: int, season: str = "2023-24",
                           games: int = 10) -> Optional[pd.DataFrame]:
        """Get player game log for recent games"""
        try:
            game_log = playergamelog.PlayerGameLog(player_id=player_id, season=season)
            df = game_log.get_data_frames()[0]
            
            if len(df) > games:
                df = df.head(games)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching player game log for player {player_id}: {e}")
            return None
    
    def get_league_averages(self, season: str = "2023-24") -> Optional[Dict[str, Any]]:
        """Get league average statistics"""
        try:
            # Get league dashboard
            dashboard = leaguedashteamstats.LeagueDashTeamStats(season=season)
            df = dashboard.get_data_frames()[0]
            
            if df.empty:
                return None
            
            # Calculate league averages
            league_averages = df.mean().to_dict()
            return league_averages
            
        except Exception as e:
            logger.error(f"Error fetching league averages: {e}")
            return None
    
    def _is_cache_valid(self, key: str) -> bool:
        """Check if cached data is still valid"""
        if key not in self.cache:
            return False
        
        cache_time, _ = self.cache[key]
        return (datetime.now() - cache_time).seconds < self.cache_timeout
    
    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Get data from cache if valid"""
        if self._is_cache_valid(key):
            _, data = self.cache[key]
            return data
        return None
    
    def _set_cache(self, key: str, data: Any) -> None:
        """Set data in cache"""
        self.cache[key] = (datetime.now(), data)
